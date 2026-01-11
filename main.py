import os
import sys
import subprocess
import json
import time
import argparse
import warnings
import torch
import functools

# --- 1. 增强版 PyTorch 2.6 兼容补丁 (必须放在最前面) ---
# 解决 Lightning 和 Torch 2.6 强制 weights_only=True 导致的模型加载失败
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
# --------------------------------------------------

# 屏蔽繁琐的警告
warnings.filterwarnings("ignore")

def install_requirements():
    """环境自愈逻辑：检查并安装缺失的第三方库"""
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
            # 使用清华源加速安装
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                install_name, 
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
            ])

def run():
    # --- 2. 延迟导入 (Lazy Import) ---
    # 确保在 install_requirements 运行后才加载第三方库
    import whisperx
    from opencc import OpenCC

    # 3. 解析命令行参数
    parser = argparse.ArgumentParser(description="WhisperX 转录工具 (CPU 稳定版)")
    parser.add_argument("--file", type=str, required=True, help="音频文件名")
    parser.add_argument("--num", type=int, default=2, help="说话人数量")
    args = parser.parse_args()

    # 4. 初始化输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_basename = os.path.basename(args.file)
    base_name_no_ext = os.path.splitext(file_basename)[0]

    # 定义文件路径
    optimized_audio = os.path.join(output_dir, f"{base_name_no_ext}_16k_mono.wav")
    cache_file = os.path.join(output_dir, f"{base_name_no_ext}_transcribe.json")
    output_txt = os.path.join(output_dir, f"{base_name_no_ext}_结果.txt")

    if not os.path.exists(args.file):
        print(f"❌ 错误：找不到文件 {args.file}")
        return

    # 5. 预处理：优化音频采样率
    if not os.path.exists(optimized_audio):
        print(f"正在优化音频采样率 (FFmpeg)...")
        # 强制转换为 16kHz 单声道以提高识别准确度
        cmd = f"ffmpeg -err_detect ignore_err -i '{args.file}' -ar 16000 -ac 1 -c:a pcm_s16le '{optimized_audio}' -y -loglevel quiet"
        subprocess.run(cmd, shell=True)

    device = "cpu"
    compute_type = "int8" 
    
    # 6. 加载模型 (Step 1/4)
    print(f">> [步骤 1/4] 加载模型...")
    model = whisperx.load_model(
        "base", 
        device, 
        compute_type=compute_type, 
        language="zh",
        asr_options={
            "beam_size": 5,
            "repetition_penalty": 1.1
        }
    )

    # 【关键修正】先加载音频数据
    audio = whisperx.load_audio(optimized_audio)

    # 7. 转录阶段 (Step 2/4)
    if not os.path.exists(cache_file):
        print(f">> [步骤 2/4] 正在转录...")
        result = model.transcribe(
            audio, 
            batch_size=4, 
            print_progress=True,
            language="zh"
        )
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    else:
        print(f">> [步骤 2/4] 读取缓存...")
        with open(cache_file, "r", encoding="utf-8") as f:
            result = json.load(f)

    # 8. 说话人识别 (Step 3/4)
    print(f">> [步骤 3/4] 分析说话人 (数量: {args.num})...")
    from whisperx import diarize
    diarize_model = diarize.DiarizationPipeline(use_auth_token=None, device=device)
    diarize_segments = diarize_model(audio, min_speakers=args.num, max_speakers=args.num)
    
    # 将说话人信息关联到文本段落
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 9. 生成结果并转换繁简 (Step 4/4)
    cc = OpenCC('t2s')
    print(f">> [步骤 4/4] ✨ 正在生成结果...")
    with open(output_txt, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker = segment.get('speaker', '未知说话人')
            # 转换繁体字为简体字
            text = cc.convert(segment['text'])
            f.write(f"[{speaker}] {text}\n")
    
    print(f"✅ 完成！最终结果已保存至: {output_txt}")

if __name__ == "__main__":
    # 启动前先自检并补齐环境
    install_requirements()
    run()