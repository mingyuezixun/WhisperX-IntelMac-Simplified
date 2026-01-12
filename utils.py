# utils.py
import torch
import warnings
import pandas as pd
import time
from opencc import OpenCC

def setup_environment():
    """环境初始化与补丁"""
    warnings.filterwarnings("ignore")
    # Torch 2.4+ 兼容性补丁
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    print("✅ [Utils] 环境补丁已加载")

def format_time(seconds):
    """辅助函数：将秒转为 H:M:S"""
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def export_results(transcript_segments, diarize_data, output_file):
    """
    [适配器模式] 核心导出函数
    无论 diarize_data 是 List 还是 DataFrame，这里都统一处理
    """
    print(f"📝 [Utils] 正在导出结果到: {output_file}")
    cc = OpenCC('t2s')
    
    # --- 1. 数据适配 ---
    # 将输入统一转换为 list of dicts 格式，方便遍历
    diarize_list = []
    if diarize_data is None:
        print("⚠️ [Utils] 警告：声纹数据为空，将全部标记为 [未知]")
    elif isinstance(diarize_data, pd.DataFrame):
        diarize_list = diarize_data.to_dict('records')
    elif isinstance(diarize_data, list):
        diarize_list = diarize_data
    else:
        print(f"❌ [Utils] 无法识别的声纹数据格式: {type(diarize_data)}")

    # --- 2. 匹配与写入 ---
    with open(output_file, "w", encoding="utf-8") as f:
        for seg in transcript_segments:
            speaker = "未知"
            mid = (seg['start'] + seg['end']) / 2
            
            # 简单暴力的循环匹配 (最稳健)
            for d in diarize_list:
                if d['start'] <= mid <= d['end']:
                    speaker = d['speaker']
                    break
            
            text = cc.convert(seg['text']).strip()
            f.write(f"[{speaker}] {text}\n")
            
    print("✅ [Utils] 导出完成！")