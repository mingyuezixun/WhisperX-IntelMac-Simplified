import os
import sys
import subprocess
import json
import time
import gc
import argparse
import warnings
import torch
import whisperx
from opencc import OpenCC


def install_requirements():
    """自动安装缺失的库"""
    libs = {
        "whisperx": "whisperx",
        "pydub": "pydub",
        "opencc-python-reimplemented": "opencc"
    }
    for install_name, import_name in libs.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"📦 正在自动安装缺失的库: {install_name}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                install_name, 
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
            ])


def setup_environment():
    # 1. 忽略不必要的警告
    warnings.filterwarnings("ignore")
    
    # 2. 解决 Torch 2.4+ 加载模型的兼容性问题
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    
    return torch
    
class WhisperProcessor:
    def __init__(self, input_file, num_speakers=2, model_size="small", chunk_mins=5):
        """
        初始化 WhisperProcessor 类
        :param input_file: 输入音频文件路径
        :param num_speakers: 说话人数量
        :param model_size: Whisper 模型大小
        :param chunk_mins: 分段切片分钟数
        """
        self.input_file = input_file
        self.num_speakers = num_speakers
        self.chunk_mins = chunk_mins
        self.model_size = model_size
        self.device = "cpu"
        self.compute_type = "int8"
        # 输出目录
        self.output_dir = "output"
        file_basename = os.path.basename(self.input_file)
        base_name_no_ext = os.path.splitext(file_basename)[0]
        # 优化后的音频路径
        self.optimized_audio = os.path.join(self.output_dir, f"{base_name_no_ext}_16k_mono.wav")
        # 缓存文件路径
        self.cache_file = os.path.join(self.output_dir, f"{base_name_no_ext}_transcribe.json")
        # 临时识别说话人的缓存路径
        self.diarize_csv = f"{base_name_no_ext}_diarize_temp.csv"
        # 最终输出文本路径
        self.output_txt = os.path.join(self.output_dir, f"{base_name_no_ext}_结果.txt")

    def pre_process(self):
        # 2. 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # 3. 优化音频采样率
        if not os.path.exists(self.optimized_audio):
            print(f"正在优化音频采样率...")
            cmd = f"ffmpeg -err_detect ignore_err -i '{self.input_file}' -ar 16000 -ac 1 -c:a pcm_s16le '{self.optimized_audio}' -y -loglevel quiet"
            subprocess.run(cmd, shell=True)


    def transcribe_audio(self):
        """
        音频转录函数
        """
        # 加载模型
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 1/4] 加载模型 ({self.model_size})...")
        model = whisperx.load_model(
            self.model_size, 
            self.device, 
            compute_type=self.compute_type, 
            language="zh",
            asr_options={"beam_size": 5}
        )

        # 加载音频
        audio = whisperx.load_audio(self.optimized_audio)

        # 转录
        if not os.path.exists(self.cache_file):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 2/4] 正在转录 (VAD优化版)...")
            result = model.transcribe(
                audio, 
                batch_size=4, 
                print_progress=True,
                language="zh",
            )
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 2/4] 读取缓存...")
            with open(self.cache_file, "r", encoding="utf-8") as f:
                result = json.load(f)
        
        # 转录完立刻手动清理内存，为声纹识别腾空间
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result, audio

    def diarize_full(self, audio_data):
        """
        不分段的全量声纹识别（对比实验用）
        """
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> ⚠️ 启动全量声纹识别（不分段）...")
        
        # 同样使用你刚才成功的加速参数
        torch.set_num_threads(4)
        pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=None, device=self.device)
        pipeline.segmentation_step = 0.5 

        start_time = time.time()
        
        try:
            # 直接对全量音频数据进行识别
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在处理全片音频（总长: {len(audio_data)/16000:.2f}s）...")
            self.diarize_df = pipeline(audio_data, min_speakers=self.num_speakers, max_speakers=self.num_speakers)
            
            elapsed = time.time() - start_time
            print(f"✅ 全量识别完成！耗时: {elapsed/60:.2f} 分钟")
        except Exception as e:
            print(f"❌ 全量识别过程中出现错误: {e}")

        return self.diarize_df

    # --- 函数 格式化生成结果 ---
    def save_final_results(self, segments):
        print(f"[{time.strftime('%H:%M:%S')}] >> 步骤 4: 导出简体中文结果...")
        cc = OpenCC('t2s')
        with open(self.output_txt, "w", encoding="utf-8") as f:
            for s in segments:
                speaker = s.get('speaker', '未知')
                text = cc.convert(s['text']).strip()
                f.write(f"[{speaker}] {text}\n")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 处理完成！结果保存在: {self.output_txt}")

    def run(self):
        # 预处理
        self.pre_process()
        # 转录音频
        result, audio = self.transcribe_audio()
        # 说话人识别
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 3/4] 分析说话人 safe_diarize...")
        diarize_segments = self.diarize_full(audio)
        # 关联说话人
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 3/4] 分析说话人 assign_speakers_simple")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 保存最终结果
        self.save_final_results(result["segments"])

def parse_args():
    """解析命令行参数 - 放在类外面，作为脚本的入口控制"""
    parser = argparse.ArgumentParser(description="WhisperX 心理咨询处理工具")
    parser.add_argument("input", help="输入音频文件路径")
    parser.add_argument("-n", "--num", type=int, default=2, help="说话人数量")
    parser.add_argument("-m", "--model", type=str, default="small", help="模型大小")
    parser.add_argument("-c", "--chunk", type=int, default=10, help="分段切片分钟数")
    return parser.parse_args()

def main():
    """主逻辑入口"""
    # 解析参数
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"❌ 找不到文件: {args.input}")
        return
    # 环境准备
    install_requirements()
    # 重新设置环境
    setup_environment()

    start_time = time.time()
    processor = WhisperProcessor(
        input_file=args.input, 
        num_speakers=args.num, 
        model_size=args.model, 
        chunk_mins=args.chunk
    )

    try:
        processor.run()
        total_time = time.time() - start_time
        print(f"\n{'-'*30}")
        print(f"⏱️ 总运行耗时: {total_time/60:.2f} 分钟")
        print(f"{'-'*30}")
    except Exception as e:
        print(f"💥 运行过程中出现错误: {e}")

if __name__ == "__main__":
    main()
