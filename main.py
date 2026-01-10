import os
import sys
import subprocess
import json
import time
import argparse
import warnings

# 屏蔽繁琐的警告
warnings.filterwarnings("ignore")

def install_requirements():
    requirements = ["whisperx", "pydub", "opencc-python-reimplemented"]
    for lib in requirements:
        try:
            __import__(lib.replace("-", "_"))
        except ImportError:
            print(f"📦 正在自动安装缺失的库: {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# 启动前检查环境
install_requirements()

import torch
import whisperx
from opencc import OpenCC

def get_optimized_audio(input_file):
    base_name = os.path.splitext(input_file)[0]
    optimized_path = f"{base_name}_16k_mono.wav"
    
    if os.path.exists(optimized_path):
        return optimized_path

    print(f"⚡ 正在优化音频采样率 (16k/单声道/修复坏帧)...")
    cmd = f"ffmpeg -err_detect ignore_err -i '{input_file}' -ar 16000 -ac 1 -c:a pcm_s16le '{optimized_path}' -y -loglevel quiet"
    subprocess.run(cmd, shell=True)
    return optimized_path

def run():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="WhisperX 转录工具")
    parser.add_argument("--file", type=str, required=True, help="音频文件名")
    parser.add_argument("--num", type=int, default=2, help="说话人数量")
    args = parser.parse_args()

    device = "cpu"
    compute_type = "int8" 
    
    if not os.path.exists(args.file):
        print(f"❌ 错误：找不到文件 {args.file}")
        return

    # 2. 预处理
    audio_file = get_optimized_audio(args.file)
    base_name = os.path.splitext(args.file)[0]
    cache_file = f"{base_name}_transcribe.json"
    output_txt = f"{base_name}_结果.txt"

    # 3. 转录阶段
    print(f">> [步骤 1/3] 加载模型...")
    model = whisperx.load_model("base", device, compute_type=compute_type, language="zh")
    audio = whisperx.load_audio(audio_file)

    if not os.path.exists(cache_file):
        print(f">> [步骤 2/3] 正在转录...")
        result = model.transcribe(audio, batch_size=4, print_progress=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    else:
        print(f">> [步骤 2/3] 读取缓存...")
        with open(cache_file, "r", encoding="utf-8") as f:
            result = json.load(f)

    # 4. 说话人识别
    print(f">> [步骤 3/3] 分析说话人 (数量: {args.num})...")
    from whisperx import diarize
    diarize_model = diarize.DiarizationPipeline(use_auth_token=None, device=device)
    diarize_segments = diarize_model(audio, min_speakers=args.num, max_speakers=args.num)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 5. 繁简转换并保存
    cc = OpenCC('t2s')
    print(f">> ✨ 正在生成结果...")
    with open(output_txt, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
            text = cc.convert(segment['text'])
            f.write(f"[{speaker}] {text}\n")
    
    print(f"✅ 完成！结果已保存至: {output_txt}")

if __name__ == "__main__":
    run()
    