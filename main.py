import os
import sys
import subprocess
import json
import time
import argparse
import warnings
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 1. PyTorch 2.6 兼容补丁 ---
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
# -----------------------------

warnings.filterwarnings("ignore")

def install_requirements():
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

def run():
    # --- 2. 延迟导入 ---
    import whisperx
    from opencc import OpenCC

    parser = argparse.ArgumentParser(description="WhisperX Intel Mac 优化版")
    parser.add_argument("--file", type=str, required=True, help="音频文件名")
    parser.add_argument("--num", type=int, default=2, help="说话人数量")
    parser.add_argument("--model", type=str, default="small", help="模型大小")
    args = parser.parse_args()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_basename = os.path.basename(args.file)
    base_name_no_ext = os.path.splitext(file_basename)[0]

    optimized_audio = os.path.join(output_dir, f"{base_name_no_ext}_16k_mono.wav")
    cache_file = os.path.join(output_dir, f"{base_name_no_ext}_transcribe.json")
    output_txt = os.path.join(output_dir, f"{base_name_no_ext}_结果.txt")

    if not os.path.exists(args.file):
        print(f"❌ 错误：找不到文件 {args.file}")
        return

    # 3. 预处理
    if not os.path.exists(optimized_audio):
        print(f"正在优化音频采样率...")
        cmd = f"ffmpeg -err_detect ignore_err -i '{args.file}' -ar 16000 -ac 1 -c:a pcm_s16le '{optimized_audio}' -y -loglevel quiet"
        subprocess.run(cmd, shell=True)

    device = "cpu"
    compute_type = "int8" 
    
    # 4. 加载模型 (Step 1/4)
    # 升级为 small 以解决说话人识别不准的问题
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 1/4] 加载模型 ({args.model})...")
    model = whisperx.load_model(
        args.model, 
        device, 
        compute_type=compute_type, 
        language="zh",
        asr_options={
            "beam_size": 5  # 只保留这个最稳的参数
        }
    )

    # 5. 加载音频 (修复了之前的 NameError)
    audio = whisperx.load_audio(optimized_audio)

    # 6. 转录 (Step 2/4)
    if not os.path.exists(cache_file):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 2/4] 正在转录 (VAD优化版)...")
        result = model.transcribe(
            audio, 
            batch_size=4, 
            print_progress=True,
            language="zh",
            # 针对指定场景优化的 VAD 参数
            # vad_options={
            #     "vad_onset": 0.5,  # 过滤电流声
            #     "vad_offset": 0.4  # 保护慢速对话不被切断
            # }
        )
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 2/4] 读取缓存...")
        with open(cache_file, "r", encoding="utf-8") as f:
            result = json.load(f)

    # 7. 说话人识别 (Step 3/4)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 3/4] 分析说话人 (强制区分 {args.num} 人)...")
    from whisperx import diarize
    diarize_model = diarize.DiarizationPipeline(use_auth_token=None, device=device)
    
    # 强制指定人数，防止两人的声音被合并
    diarize_segments = diarize_model(
        audio, 
        min_speakers=args.num, 
        max_speakers=args.num
    )
    
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 8. 生成结果
    cc = OpenCC('t2s')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] >> [步骤 4/4] ✨ 正在生成结果...")
    with open(output_txt, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker = segment.get('speaker', '未知说话人')
            text = cc.convert(segment['text'])
            f.write(f"[{speaker}] {text}\n")
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 完成！结果已保存至: {output_txt}")

if __name__ == "__main__":
    install_requirements()
    run()