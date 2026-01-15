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
        
        # 2. 说话人识别 (分段执行 Logic mapped from previous SegmentedDiarization)
        logger.info(">> Step 2: Segmented Diarization...")
        audio = whisperx.load_audio(audio_path)
        
        total_sec = len(audio) / 16000
        chunk_sec = self.chunk_mins * 60
        all_chunks = []
        min_speakers = config.get("min_speakers")
        max_speakers = config.get("max_speakers")

        for i, start_s in enumerate(range(0, int(total_sec), chunk_sec)):
            end_s = min(start_s + chunk_sec, total_sec)
            logger.info(f"   -> Diarize Chunk {i+1}: {int(start_s//60)}-{int(end_s//60)} min")
            
            # Slice audio directly from loaded array
            chunk_audio = audio[int(start_s * 16000) : int(end_s * 16000)]
            
            try:
                # Run diarization on chunk
                df = diarize_model(
                    chunk_audio, 
                    min_speakers=min_speakers, 
                    max_speakers=max_speakers
                )
                
                # Use standard pandas functionality to avoid copy warnings if needed, 
                # but direct reassignment is usually fine here on fresh DF
                df['start'] += start_s
                df['end'] += start_s
                
                all_chunks.append(df)
            except Exception as e:
                logger.error(f"    Chunk {i+1} Failed: {e}")
            
            gc.collect()

        diarize_data = None
        if all_chunks:
            diarize_data = pd.concat(all_chunks, ignore_index=True)

        # 3. 【核心逻辑】挂载数据，留给 Processor 中的 export_results 处理
        result["diarize_data"] = diarize_data
        
        return result
