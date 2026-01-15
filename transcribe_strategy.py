# transcribe_strategy.py
from abc import ABC, abstractmethod
import whisperx
import torch
import pandas as pd
import gc
import os
import json
from logger import logger
from utils import export_results

class TranscribeStrategy(ABC):
    @abstractmethod
    def process(self, audio_path, model, diarize_model, config, cache_file=None):
        """
        执行转录 + 说话人匹配的核心逻辑
        Args:
            audio_path: 音频文件路径
            model: 已加载的 Whisper 模型
            diarize_model: 已加载的 Diarization Pipeline
            config: 配置字典
            cache_file: (Optional) 转录结果缓存路径
        Returns:
            result: 包含 segments 和 potential diarize_data 的字典
        """
        pass

    def _transcribe_with_cache(self, model, audio_path, config, cache_file):
        """辅助方法：带缓存的转录"""
        if cache_file and os.path.exists(cache_file):
            logger.info(f">> Loading Transcription from Cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        logger.info(">> Step 1: Transcribing...")
        result = model.transcribe(
            audio_path, 
            batch_size=config.get("batch_size", 4), 
            language=config.get("language", "zh"),
            print_progress=True,
        )
        
        # Save cache if path provided
        if cache_file:
            logger.info(f">> Saving Transcription Cache to {cache_file}")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                
        return result

class FullStrategy(TranscribeStrategy):
    def process(self, audio_path, model, diarize_model, config, cache_file=None):
        device = config.get("device", "cuda")
        logger.info("[Strategy] Mode: Full - 启用 align 和 assign_word_speakers")
        
        # 1. 转录 (带缓存)
        result = self._transcribe_with_cache(model, audio_path, config, cache_file)
        
        # 2. 对齐 (assign_word_speakers 的前置必须步骤)
        logger.info(">> Step 2: Aligning...")

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=device
        )
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio_path, 
            device, 
            return_char_alignments=False
        )
        
        # 3. 说话人识别 (Diarization)
        logger.info(">> Step 3: Diarization...")
        
        diarize_segments = diarize_model(
            audio_path, 
            min_speakers=config.get("min_speakers"), 
            max_speakers=config.get("max_speakers")
        )
        
        # 4. 【核心逻辑】使用官方方法匹配说话人
        logger.info(">> Step 4: Assigning Speakers...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 释放显存
        del model_a
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result

class SegmentStrategy(TranscribeStrategy):
    def __init__(self, chunk_mins=10):
        self.chunk_mins = chunk_mins

    def process(self, audio_path, model, diarize_model, config, cache_file=None):
        logger.info(f"[Strategy] Mode: Segment - 自定义分段 (Chunk {self.chunk_mins} min)")
        
        # 1. 转录 (带缓存)
        result = self._transcribe_with_cache(model, audio_path, config, cache_file)
        
        # 2. 准备 Align 模型 (新增：为了获得 Word-Level 时间戳)
        device = config.get("device", "cuda")
        logger.info(">> Loading Alignment Model for Segment Mode...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=device
        )

        # 3. 分段处理 (Alignment + Diarization + Assign)
        logger.info(">> Step 2: Segmented Processing (Align + Diarize + Assign)...")
        audio = whisperx.load_audio(audio_path)
        
        total_sec = len(audio) / 16000
        chunk_sec = self.chunk_mins * 60
        all_segments = [] # 存储处理后的所有 segments (带 speaker 标签)
        
        min_speakers = config.get("min_speakers")
        max_speakers = config.get("max_speakers")

        for i, start_s in enumerate(range(0, int(total_sec), chunk_sec)):
            end_s = min(start_s + chunk_sec, total_sec)
            logger.info(f"   -> Processing Chunk {i+1}: {int(start_s//60)}-{int(end_s//60)} min")
            
            # A. 提取本段音频
            chunk_audio = audio[int(start_s * 16000) : int(end_s * 16000)]
            
            # B. 筛选属于本段的 Transcription Segments
            chunk_transcription = {"segments": [], "language": result["language"]}
            for seg in result["segments"]:
                # 如果很多 seg 跨越了边界，这里可能需要更复杂的逻辑
                # 但简单起见，只要中心点在范围内就算
                mid = (seg['start'] + seg['end']) / 2
                if start_s <= mid < end_s:
                    # 必须把时间偏移量减掉，Align模型才能对齐 (因为输入的是切片音频)
                    seg_copy = seg.copy()
                    seg_copy['start'] -= start_s
                    seg_copy['end'] -= start_s
                    chunk_transcription["segments"].append(seg_copy)
            
            if not chunk_transcription["segments"]:
                continue

            try:
                # C. Align (获取字级时间戳)
                aligned_result = whisperx.align(
                    chunk_transcription["segments"], 
                    model_a, 
                    metadata, 
                    chunk_audio, 
                    device, 
                    return_char_alignments=False
                )
                
                # D. Diarize
                diarize_segments = diarize_model(
                    chunk_audio, 
                    min_speakers=min_speakers, 
                    max_speakers=max_speakers
                )
                
                # E. Assign Speakers (核心优化)
                final_chunk_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
                
                # F. 恢复时间偏移量
                for seg in final_chunk_result["segments"]:
                    seg['start'] += start_s
                    seg['end'] += start_s
                    if "words" in seg:
                        for word in seg["words"]:
                            if "start" in word: word['start'] += start_s
                            if "end" in word: word['end'] += start_s
                            
                all_segments.extend(final_chunk_result["segments"])

            except Exception as e:
                logger.error(f"    Chunk {i+1} Failed: {e}")
                # 降级：保留原始 transcription，标记为未知
                for seg in chunk_transcription["segments"]:
                     seg['start'] += start_s
                     seg['end'] += start_s
                     all_segments.append(seg)
            
            gc.collect()

        # 4. 替换 result["segments"] 为带有 speaker 信息的 segments
        result["segments"] = sorted(all_segments, key=lambda x: x['start'])
        
        # 释放资源
        del model_a
        return result
