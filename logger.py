# logger.py
import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

# 定义全局对象
logger = logging.getLogger("WhisperGlobal")

class ShortNameFormatter(logging.Formatter):
    """自定义格式化工具：缩写长文件名"""
    def format(self, record):
        # 如果文件名超过 15 个字符，只取后 12 位并加前缀 ~
        max_len = 15
        if len(record.filename) > max_len:
            record.short_filename = "~" + record.filename[-(max_len-1):]
        else:
            record.short_filename = record.filename
        return super().format(record)


def init_global_config(log_dir, level_str="INFO"):
    """
    [由 main 调用] 全局保底配置：仅设置级别和控制台输出
    """
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    if not logger.handlers:
        log_format = '%(asctime)s [%(levelname)s] [%(short_filename)s:%(lineno)d] %(message)s'
        formatter = ShortNameFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

        # --- A. 控制台输出 ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # --- B. 文件输出 (核心修复) ---
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        global_log_path = os.path.join(log_dir, "main.log")
        try:
            # when="midnight": 每天凌晨触发滚动
            # interval=1: 间隔为 1 天
            # backupCount=15: 只保留最近 15 天的日志，旧的自动删除
            file_handler = TimedRotatingFileHandler(
                filename=global_log_path,
                when="midnight",
                interval=1,
                backupCount=7,
                encoding='utf-8'
            )
            
            # 设置后缀格式，让旧日志文件名变成 main.log.2026-01-13
            file_handler.suffix = "%Y-%m-%d"
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"💾 日志文件已挂载: {global_log_path}")
        except Exception as e:
            # 如果文件写不了，至少控制台能看到报错
            print(f"❌ 无法创建日志文件: {e}")
    
    logger.info(f"🌟 全局日志系统已启动 (Level: {level_str.upper()})")

def get_sub_logger(name, log_dir=None, filename=None):
    """
    [按需调用] 获取子日志器，可选是否挂载文件
    """
    sub_logger = logging.getLogger(f"WhisperGlobal.{name}")
    
    # 如果指定了目录和文件名，则挂载文件输出
    if log_dir and filename:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, filename)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        sub_logger.addHandler(file_handler)
        
    return sub_logger