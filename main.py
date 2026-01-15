# main.py
import argparse
import os
import json
import utils
from logger import logger, init_global_config
from processor import WhisperProcessor
from transcribe_strategy import FullStrategy, SegmentStrategy

def parse_args(conf):
    parser = argparse.ArgumentParser(description="WhisperX 模块化处理工具")
    parser.add_argument("input", default=conf.get("input", ""), help="输入音频文件")
    parser.add_argument("--mode", choices=["full", "segment"], default=conf.get("mode", "full"), help="识别模式：full(全量) 或 segment(分段)")
    parser.add_argument("-m", "--model", type=str, default=conf.get("model", "small"), help="模型大小")
    parser.add_argument("-n", "--num", type=int, default=conf.get("num_speakers", None), help="说话人数 (固定)")
    parser.add_argument("--min", type=int, default=conf.get("min_speakers", None), help="最少说话人数")
    parser.add_argument("--max", type=int, default=conf.get("max_speakers", None), help="最多说话人数")
    parser.add_argument("--hf_token", type=str, default=conf.get("hf_token", None), help="HuggingFace Token")
    parser.add_argument("--vad_onset", type=float, default=conf.get("vad_onset", 0.500), help="VAD Onset Threshold")
    parser.add_argument("--vad_offset", type=float, default=conf.get("vad_offset", 0.363), help="VAD Offset Threshold")
    parser.add_argument("--vad_min_duration_on", type=float, default=conf.get("vad_min_duration_on", 0.1), help="VAD 最小语音时长")
    parser.add_argument("--vad_min_duration_off", type=float, default=conf.get("vad_min_duration_off", 0.1), help="VAD 最小静音时长")
    parser.add_argument("-c", "--chunk", type=int, default=conf.get("chunk", 10), help="分段时长(分钟)，仅在 segment 模式有效")
    parser.add_argument("-l", "--language", type=str, default=conf.get("language", "zh"), help="语言代码，如 zh, en")
    return parser.parse_args()

def init(conf):
    utils.ensure_dir(conf.get("logs_dir", "./logs"))
    utils.install_requirements()
    utils.setup_environment()
        
def main():
    # 读取配置文件
    conf = utils.load_config()
    args = parse_args(conf)
    init(conf)

    # 初始化全局日志
    init_global_config(log_dir=conf.get("logs_dir", "./logs"), level_str=conf.get("log_level", "INFO"))
    
    # 打印配置文件内容
    logger.info("=" * 50)
    logger.info(f"Config File: {json.dumps(conf, indent=4, ensure_ascii=False)}")
    
    # 打印命令行参数内容
    logger.info(f"Command Args: {json.dumps(vars(args), indent=4, ensure_ascii=False)}")
    logger.info("=" * 50)

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return
    
    # --- 策略模式：决定使用哪种策略 ---
    if args.mode == "full":
        strategy = FullStrategy()
    else:
        # Segment 模式
        strategy = SegmentStrategy(chunk_mins=args.chunk)
    # ----------------------------------

    # 说话人参数逻辑
    min_speakers = args.min
    max_speakers = args.max
    # 如果指定了固定人数，优先级高于 min/max
    if args.num is not None:
        min_speakers = args.num
        max_speakers = args.num

    # 配置参数
    config = {
        "model": args.model,
        "device": conf.get("device", "cpu"),
        "num_speakers": args.num, # 保留以备不时之需，但主要用 min/max
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "hf_token": args.hf_token,
        "vad_onset": args.vad_onset,
        "vad_offset": args.vad_offset,
        "vad_min_duration_on": args.vad_min_duration_on,
        "vad_min_duration_off": args.vad_min_duration_off,
        "threads": conf.get("threads", 4),
        "output": conf.get("output", "./outputs"),
        "compute_type": conf.get("compute_type", "int8"),
        "batch_size": conf.get("batch_size", 4),
        "language": args.language,
        "mode": args.mode, 
    }
    # 实例化并运行
    processor = WhisperProcessor(args, strategy, config)
    processor.run()

if __name__ == "__main__":
    main()