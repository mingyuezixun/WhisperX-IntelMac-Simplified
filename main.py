# main.py
import argparse
import os
from processor import WhisperProcessor
from strategies import FullDiarization, SegmentedDiarization

def parse_args():
    parser = argparse.ArgumentParser(description="WhisperX 模块化处理工具")
    parser.add_argument("input", help="输入音频文件")
    parser.add_argument("--mode", choices=["full", "segment"], default="full", help="识别模式：full(全量) 或 segment(分段)")
    parser.add_argument("-m", "--model", type=str, default="small", help="模型大小")
    parser.add_argument("-n", "--num", type=int, default=2, help="说话人数")
    parser.add_argument("-c", "--chunk", type=int, default=10, help="分段时长(分钟)，仅在 segment 模式有效")
    parser.add_argument("--output-dir", dest="output", default="outputs", help="转录结果存放目录")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return

    # --- 简单工厂模式：决定使用哪种策略 ---
    if args.mode == "full":
        strategy = FullDiarization()
    else:
        strategy = SegmentedDiarization(chunk_mins=args.chunk)
    # ----------------------------------

    # 配置参数
    config = {
        "model": args.model,
        "device": "cpu",
        "num_speakers": args.num,
        "threads": 4
    }

    # 实例化并运行
    processor = WhisperProcessor(args, strategy, config)
    processor.run()

if __name__ == "__main__":
    main()