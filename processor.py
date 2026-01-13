# processor.py
import whisperx
import torch
import os
import gc
import utils
from logger import logger
from utils import export_results
import subprocess

class WhisperProcessor:
    def __init__(self, args, strategy, config):
        self.input_file = args.input
        self.strategy = strategy # 接收 TranscribeStrategy 实例
        self.config = config     # 参数字典
        self.output_dir = self.config['output']

        file_basename = os.path.basename(self.input_file)
        base_name_no_ext = os.path.splitext(file_basename)[0]

        # 最终输出文本路径
        self.output_txt = os.path.join(self.output_dir, f"{base_name_no_ext}_final.txt")
        # 优化后的音频路径
        self.optimized_audio = os.path.join(self.output_dir, f"{base_name_no_ext}_16k_mono.wav")
        # 缓存文件路径: {filename}_{model}_transcribe.json
        model_name = self.config.get('model', 'unknown')
        self.cache_file = os.path.join(self.output_dir, f"{base_name_no_ext}_{model_name}_transcribe.json")
        
        self.model = None
        self.diarize_model = None

        # 初始化目录
        self._init_directories()

    def _init_directories(self):
        # 确保根输出目录存在
        utils.ensure_dir(self.output_dir)
        # 针对当前文件创建独立的子目录 (可选，保持原逻辑)
        # file_name = os.path.splitext(os.path.basename(self.input_file))[0]
        # self.task_dir = os.path.join(self.output_dir, file_name)
        # utils.ensure_dir(self.task_dir)

    def load_resource(self):
        # 1. 优化音频
        if not os.path.exists(self.optimized_audio):
            logger.info(f"正在优化音频采样率...")
            cmd = f"ffmpeg -err_detect ignore_err -i '{self.input_file}' -ar 16000 -ac 1 -c:a pcm_s16le '{self.optimized_audio}' -y -loglevel quiet"
            subprocess.run(cmd, shell=True)

        # 2. 加载 Whisper 模型
        logger.info("[Processor] Loading Whisper Model...")
        self.model = whisperx.load_model(
            self.config['model'], 
            self.config['device'], 
            compute_type=self.config['compute_type'],
            language=self.config['language'],
            asr_options={"beam_size": 5}
        )

        # 3. 加载 Diarization 模型
        logger.info("[Processor] Loading Diarization Pipeline...")
        # 配置线程
        torch.set_num_threads(self.config['threads'])
        self.diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=None,  # 假设用户已经登录或不需要 token
            device=self.config['device']
        )
        self.diarize_model.segmentation_step = 0.5

    def run(self):
        try:
            # 1. 准备资源
            self.load_resource()
            
            # 2. 运行策略 (包含转录和说话人匹配)
            # 注意：传入 optimized_audio 和 cache_file
            result = self.strategy.process(
                self.optimized_audio, 
                self.model, 
                self.diarize_model, 
                self.config,
                cache_file=self.cache_file
            )
            
            # 3. 差异化导出
            mode = self.config.get("mode", "full") # config 中应该包含 mode，或者这里判断策略类型
            
            # 此处判断一下 config 中的 mode，或者根据 result 特征判断
            # 如果是 Full 模式，result['segments'] 里的 word 应该包含 speaker
            # 如果是 Segment 模式，result['diarize_data'] 存在
            
            logger.info(f"执行导出 (Mode: {mode})...")
            
            if mode == "full":
                # Full 模式：直接从 result['segments'] 里读 speaker 字段
                with open(self.output_txt, "w", encoding="utf-8") as f:
                    for seg in result["segments"]:
                        spk = seg.get("speaker", "未知")
                        text = seg["text"].strip()
                        f.write(f"[{spk}] {text}\n")
            else:
                # Segment 模式：调用自定义导出函数
                export_results(
                    result["segments"], 
                    result.get("diarize_data"), 
                    self.output_txt
                )
                
            logger.info(f"[Processor] Done! Output: {self.output_txt}")

        finally:
            # 清理资源
            if self.model:
                del self.model
            if self.diarize_model:
                del self.diarize_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()