# utils.py
import torch
import warnings
import pandas as pd
import time
from logger import logger
import sys
import subprocess
from opencc import OpenCC

import sys
import subprocess
import warnings
import torch
import yaml
import os
from logger import logger  # 引用全局单例

def install_requirements():
    """
    [严格保留] 自动安装缺失的库，使用清华源镜像
    """
    libs = {
        "whisperx": "whisperx",
        "pydub": "pydub",
        "opencc-python-reimplemented": "opencc"
    }
    for install_name, import_name in libs.items():
        try:
            __import__(import_name)
        except ImportError:
            logger.info(f"正在自动安装缺失的库: {install_name}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    install_name, 
                    "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
                ])
            except Exception as e:
                logger.error(f"安装 {install_name} 失败: {e}")

def setup_environment():
    """
    [严格保留] 解决 Torch 2.4+ 加载模型的兼容性问题 (weights_only=False)
    """
    # 1. 忽略不必要的警告
    warnings.filterwarnings("ignore")
    
    # 2. 补丁：解决 Torch 权重加载限制
    logger.info("正在应用 Torch.load 兼容性补丁 (weights_only=False)...")
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        # 强制设置 weights_only 为 False
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    return torch

def load_config(config_path="conf/config.yml"):
    """
    [固定逻辑] 从 yml 读取配置
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"配置文件读取失败: {e}")
    return {}

def ensure_dir(path):
    """
    [通用工具] 确保目录存在
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def format_time(seconds):
    """辅助函数：将秒转为 H:M:S"""
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def export_results(transcript_segments, diarize_data, output_file):
    """
    [适配器模式] 核心导出函数
    无论 diarize_data 是 List 还是 DataFrame，这里都统一处理
    """
    logger.info(f"[Utils] 正在导出结果到: {output_file}")
    cc = OpenCC('t2s')
    
    # --- 1. 数据适配 ---
    # 将输入统一转换为 list of dicts 格式，方便遍历
    diarize_list = []
    if diarize_data is None:
        logger.info("[Utils] 警告：声纹数据为空，将全部标记为 [未知]")
    elif isinstance(diarize_data, pd.DataFrame):
        diarize_list = diarize_data.to_dict('records')
    elif isinstance(diarize_data, list):
        diarize_list = diarize_data
    else:
        logger.info(f"[Utils] 无法识别的声纹数据格式: {type(diarize_data)}")

    # --- 2. 匹配与写入 ---
    with open(output_file, "w", encoding="utf-8") as f:
        for seg in transcript_segments:
            speaker = "未知" # 默认值
            mid = (seg['start'] + seg['end']) / 2
            
            # 改进：如果中点没匹配上，找一个离这段话最近的说话人
            best_match = None
            min_dist = float('inf')

            for d in diarize_list:
                if d['start'] <= mid <= d['end']:
                    speaker = d['speaker']
                    break
                # 记录最近的说话人（备选）
                dist = min(abs(mid - d['start']), abs(mid - d['end']))
                if dist < min_dist:
                    min_dist = dist
                    best_match = d['speaker']
            
            # 如果没匹配上，且距离最近的说话人小于 1 秒，就借用他的 ID
            if speaker == "未知" and min_dist < 1.0:
                speaker = best_match

            text = cc.convert(seg['text']).strip()
            # 只有当 text 真的有内容时才写入
            if text:
                f.write(f"[{speaker}] {text}\n")
            
    logger.info("[Utils] 导出完成！")