# strategies.py
from abc import ABC, abstractmethod
import pandas as pd
import gc
from logger import logger

# --- 抽象基类 (接口) ---
class DiarizationStrategy(ABC):
    @abstractmethod
    def run(self, pipeline, audio_data, num_speakers):
        """必须实现 run 方法，返回声纹数据"""
        pass

@abstractmethod
    def export(self, result, output_file):
        pass

# --- 策略 A: 全量识别 (简单粗暴) ---
class FullDiarization(DiarizationStrategy):
    def run(self, pipeline, audio_data, num_speakers):
        logger.info("[Strategy] 正在执行【全量】声纹识别...")
        try:
            # 直接调用 pipeline，返回的是 List
            result = pipeline(audio_data, min_speakers=num_speakers, max_speakers=num_speakers)
            return result
        except Exception as e:
            logger.error(f"[Strategy] 全量识别失败: {e}")
            return None

# --- 策略 B: 分段识别 (稳健抗压) ---
class SegmentedDiarization(DiarizationStrategy):
    def __init__(self, chunk_mins=10):
        self.chunk_mins = chunk_mins

    def run(self, pipeline, audio_data, num_speakers):
        logger.info(f"[Strategy] 正在执行【分段】声纹识别 (每段 {self.chunk_mins} min)...")
        
        total_sec = len(audio_data) / 16000
        chunk_sec = self.chunk_mins * 60
        all_chunks = []

        for i, start_s in enumerate(range(0, int(total_sec), chunk_sec)):
            end_s = min(start_s + chunk_sec, total_sec)
            logger.info(f"   -> 处理分段 {i+1}: {int(start_s//60)}-{int(end_s//60)} min")
            
            chunk_audio = audio_data[int(start_s * 16000) : int(end_s * 16000)]
            try:
                # 识别当前段
                df = pipeline(chunk_audio, min_speakers=num_speakers, max_speakers=num_speakers)
                
                # 时间补偿
                df['start'] += start_s
                df['end'] += start_s
                # 加上后缀防止重名混淆 (可选)
                # df['speaker'] = df['speaker'] + f"_P{i+1}" 
                
                all_chunks.append(df)
            except Exception as e:
                logger.error(f"    分段 {i+1} 出错: {e}")
            
            gc.collect() # 关键：每段清理内存

        if all_chunks:
            return pd.concat(all_chunks, ignore_index=True)
        return None