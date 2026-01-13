# processor.py
import whisperx
import torch
import os
import json
import gc
import utils
from logger import logger
import subprocess
from utils import export_results

class WhisperProcessor:
    def __init__(self, args, strategy, config):
        self.input_file = args.input
        self.strategy = strategy # 这里接收一个策略对象
        self.config = config     # 参数字典 (num, model, device等)
        self.output_dir = self.config['output']

        file_basename = os.path.basename(self.input_file)
        base_name_no_ext = os.path.splitext(file_basename)[0]

        # 最终输出文本路径
        self.output_txt = os.path.join(self.output_dir, f"{base_name_no_ext}_final.txt")
        # 优化后的音频路径
        self.optimized_audio = os.path.join(self.output_dir, f"{base_name_no_ext}_16k_mono.wav")
        # 缓存文件路径
        self.cache_file = os.path.join(self.output_dir, f"{base_name_no_ext}_transcribe.json")

        self.audio = None
        self.transcript_result = None
        self.diarize_result = None

        # 初始化目录
        self._init_directories()

    def _init_directories(self):
        """
        内部初始化方法，确保输出结构完整
        """
        # 确保根输出目录存在 (如: outputs/)
        utils.ensure_dir(self.output_dir)

        # 针对当前文件创建独立的子目录 (如: outputs/test_audio/)
        file_name = os.path.splitext(os.path.basename(self.input_file))[0]
        
        # 关键：创建任务目录，这样后续的缓存和结果都有家可归
        self.task_dir = os.path.join(self.output_dir, file_name)
        utils.ensure_dir(self.task_dir)

        # 建议：创建一个隐藏的缓存目录，专门放分段、VAD 等临时文件
        self.cache_dir = os.path.join(self.task_dir, ".cache")
        utils.ensure_dir(self.cache_dir)

    def load_resource(self):
        # 优化音频采样率
        if not os.path.exists(self.optimized_audio):
            logger.info(f"正在优化音频采样率...")
            cmd = f"ffmpeg -err_detect ignore_err -i '{self.input_file}' -ar 16000 -ac 1 -c:a pcm_s16le '{self.optimized_audio}' -y -loglevel quiet"
            subprocess.run(cmd, shell=True)

        logger.info("[Processor] 加载音频...")
        self.audio = whisperx.load_audio(self.optimized_audio)

    def transcribe(self):
        """
        transcribe 的 Docstring
        转录音频为文本
        """
        logger.info("[Processor] 开始转录文本...")
        model = whisperx.load_model(
            self.config['model'], 
            self.config['device'], 
            compute_type=self.config['compute_type'],
            language=self.config['language'],
            asr_options={"beam_size": 5}
        )

        if not os.path.exists(self.cache_file):
            logger.info(">> 正在转录 (VAD优化版)...")
            self.transcript_result = model.transcribe(
                self.audio, 
                batch_size=self.config['batch_size'], 
                print_progress=True,
                language=self.config['language'],
            )
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.transcript_result, f, ensure_ascii=False, indent=4)
        else:
            logger.info(">> 读取转录缓存...")
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.transcript_result = json.load(f)
        
        # 转录完立刻手动清理内存，为声纹识别腾空间
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    

    def diarize(self):
        logger.info("[Processor] 开始声纹识别...")
        
        # 配置线程 (Intel Mac 优化)
        torch.set_num_threads(self.config['threads'])
        
        # 初始化 Pipeline
        pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=None, device=self.config['device'])
        pipeline.segmentation_step = 0.5
        # 【关键点】这里调用策略对象的 run 方法，Processor 不关心具体怎么跑
        self.diarize_result = self.strategy.run(
            pipeline, 
            self.audio, 
            self.config['num_speakers']
        )
        
        if self.diarize_result is None:
            logger.info("[Processor] 严重错误：声纹识别未返回任何数据！")

    def export(self):
        if self.transcript_result:
            export_results(
                self.transcript_result["segments"], 
                self.diarize_result, 
                self.output_txt
            )



    def run(self):
        # 加载并优化资源
        self.load_resource()
        # 识别文字
        self.transcribe()
        # 识别声纹，知道哪个时间段是谁在说话
        self.diarize()
        # 导出
        self.export()