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
    # 建立“安装名”到“导入名”的映射
    libs = {
        "whisperx": "whisperx",
        "pydub": "pydub",
        "opencc-python-reimplemented": "opencc"  # 注意这里：安装是这个，导入是 opencc
    }
    
    for install_name, import_name in libs.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"📦 正在自动安装缺失的库: {install_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# 启动前检查环境
install_requirements()

import torch
import whisperx
from opencc import OpenCC

def run():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="WhisperX 转录工具")
    parser.add_argument("--file", type=str, required=True, help="音频文件名")
    parser.add_argument("--num", type=int, default=2, help="说话人数量")
    args = parser.parse_args()

    # 2. 初始化输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取不带路径的文件名
    file_basename = os.path.basename(args.file)
    base_name_no_ext = os.path.splitext(file_basename)[0]

    # 定义输出文件的完整路径
    optimized_audio = os.path.join(output_dir, f"{base_name_no_ext}_16k_mono.wav")
    cache_file = os.path.join(output_dir, f"{base_name_no_ext}_transcribe.json")
    output_txt = os.path.join(output_dir, f"{base_name_no_ext}_结果.txt")

    if not os.path.exists(args.file):
        print(f"❌ 错误：找不到文件 {args.file}")
        return

    # 3. 预处理 (修改 ffmpeg 输出路径)
    if not os.path.exists(optimized_audio):
        print(f"正在优化音频采样率...")
        cmd = f"ffmpeg -err_detect ignore_err -i '{args.file}' -ar 16000 -ac 1 -c:a pcm_s16le '{optimized_audio}' -y -loglevel quiet"
        subprocess.run(cmd, shell=True)

    device = "cpu"
    compute_type = "int8" 
    
    # 4. 转录阶段
    print(f">> [步骤 1/4] 加载模型...")
    model = whisperx.load_model("base", device, compute_type=compute_type, language="zh")
    audio = whisperx.load_audio(optimized_audio)

    if not os.path.exists(cache_file):
        print(f">> [步骤 2/4] 正在转录...")
        result = model.transcribe(audio, batch_size=4, print_progress=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    else:
        print(f">> [步骤 2/4] 读取缓存...")
        with open(cache_file, "r", encoding="utf-8") as f:
            result = json.load(f)

    # 4. 说话人识别
    print(f">> [步骤 3/4] 分析说话人 (数量: {args.num})...")
    from whisperx import diarize
    diarize_model = diarize.DiarizationPipeline(use_auth_token=None, device=device)
    diarize_segments = diarize_model(audio, min_speakers=args.num, max_speakers=args.num)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 5. 繁简转换并保存
    cc = OpenCC('t2s')
    print(f">> [步骤 4/4] ✨ 正在生成结果...")
    with open(output_txt, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
            text = cc.convert(segment['text'])
            f.write(f"[{speaker}] {text}\n")
    
    print(f"✅ 完成！结果已保存至: {output_txt}")

if __name__ == "__main__":
    run()
    