import parselmouth
from parselmouth.praat import call
import tgt
from pathlib import Path
from typing import Optional, List


class AudioTransformer:
    """
    音频变换器：根据目标TextGrid的时间轴，将源音频进行变速对齐
    注意：所有间隙均使用纯静音填充
    """
    
    def __init__(self, min_gap_threshold: float = 0.001, 
                 min_stretch: float = 0.1, 
                 max_stretch: float = 8.0,
                 short_audio_threshold: float = 0.05,
                 verbose: bool = True):
        self.min_gap_threshold = min_gap_threshold
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch
        self.short_audio_threshold = short_audio_threshold
        self.verbose = verbose
    
    def create_silence(self, duration: float, sample_rate: int) -> Optional[parselmouth.Sound]:
        """创建纯静音"""
        if duration <= self.min_gap_threshold:
            return None
        return call("Create Sound from formula", "silence", 1, 0, duration, sample_rate, "0")
    
    def extract_audio(self, sound: parselmouth.Sound, start: float, end: float) -> Optional[parselmouth.Sound]:
        """提取音频片段"""
        if end <= start:
            return None
        return sound.extract_part(from_time=start, to_time=end, preserve_times=False)
    
    def time_stretch(self, sound: parselmouth.Sound, stretch_factor: float) -> Optional[parselmouth.Sound]:
        """智能时间拉伸 (PSOLA + Resample fallback)"""
        if sound is None:
            return None
        
        stretch_factor = max(self.min_stretch, min(stretch_factor, self.max_stretch))
        
        # 处理极短片段
        if sound.duration < self.short_audio_threshold:
            target_sr = sound.sampling_frequency / stretch_factor
            temp_sound = sound.copy()
            temp_sound.override_sampling_frequency(target_sr)
            resampled = temp_sound.resample(new_frequency=sound.sampling_frequency)
            return resampled
        
        # 处理正常片段
        try:
            manipulated = call(sound, "To Manipulation", 0.01, 75, 600)
            duration_tier = call(manipulated, "Extract duration tier")
            
            end_time = call(duration_tier, "Get end time")
            call(duration_tier, "Remove points between", 0.0, end_time)
            call(duration_tier, "Add point", 0.0, stretch_factor)
            
            call([manipulated, duration_tier], "Replace duration tier")
            stretched = call(manipulated, "Get resynthesis (overlap-add)")
            return stretched
            
        except Exception as e:
            if self.verbose:
                print(f"PSOLA failed ({e}), falling back to resampling...")
            target_sr = sound.sampling_frequency / stretch_factor
            temp_sound = sound.copy()
            temp_sound.override_sampling_frequency(target_sr)
            return temp_sound.resample(new_frequency=sound.sampling_frequency)
    
    def get_real_words(self, tier: tgt.IntervalTier) -> List[tgt.Interval]:
        """提取实词列表"""
        return [i for i in tier if i.text not in ['', 'sp', 'sil', '<eps>']]
    
    def transform(self, 
                  source_audio_path: str,
                  source_textgrid_path: str,
                  target_textgrid_path: str,
                  target_audio_path: str,
                  output_path: str,
                  tier_name: str = "words") -> bool:
        try:
            # 1. 加载资源
            if self.verbose:
                print(f"Loading resources...")
            
            tg_source = tgt.io.read_textgrid(source_textgrid_path)
            tg_target = tgt.io.read_textgrid(target_textgrid_path)
            sound_source = parselmouth.Sound(source_audio_path)
            sound_target_ref = parselmouth.Sound(target_audio_path)
            
            target_total_duration = sound_target_ref.duration
            original_sr = sound_source.sampling_frequency
            
            # 2. 提取实词列表
            tier_source = tg_source.get_tier_by_name(tier_name)
            tier_target = tg_target.get_tier_by_name(tier_name)
            
            words_source = self.get_real_words(tier_source)
            words_target = self.get_real_words(tier_target)
            
            if len(words_source) != len(words_target):
                print(f"Error: 单词数量不匹配! Source:{len(words_source)} vs Target:{len(words_target)}")
                return False
            
            final_segments = []
            last_end_target = 0.0  # 我们现在只需要跟踪目标的时间轴
            
            # 3. 处理每个单词
            for i in range(len(words_target)):
                word_source = words_source[i]
                word_target = words_target[i]
                
                # ==========================================
                # A. 处理词前间隙 (Pre-Word Gap) - 全部用静音
                # ==========================================
                gap_duration_target = word_target.start_time - last_end_target
                
                # 只要 B 需要间隙，直接生成静音，不看 A
                if gap_duration_target > self.min_gap_threshold:
                    silence_segment = self.create_silence(gap_duration_target, original_sr)
                    if silence_segment:
                        final_segments.append(silence_segment)
                
                # ==========================================
                # B. 处理单词本身 (Word) - 拉伸 A 的单词
                # ==========================================
                duration_target = word_target.end_time - word_target.start_time
                duration_source = word_source.end_time - word_source.start_time
                
                # 提取 A 的单词内容
                raw_word_source = self.extract_audio(sound_source, word_source.start_time, word_source.end_time)
                
                if duration_target > self.min_gap_threshold and raw_word_source:
                    stretch_ratio = duration_target / duration_source
                    processed_word = self.time_stretch(raw_word_source, stretch_ratio)
                    final_segments.append(processed_word)
                
                # 更新指针
                last_end_target = word_target.end_time
            
            # ==========================================
            # 4. 处理尾部间隙 (Tail Gap) - 全部用静音
            # ==========================================
            tail_gap_target = target_total_duration - last_end_target
            
            if tail_gap_target > self.min_gap_threshold:
                tail_silence = self.create_silence(tail_gap_target, original_sr)
                if tail_silence:
                    final_segments.append(tail_silence)
            
            # 5. 拼接与保存
            if not final_segments:
                print("Error: 未生成音频片段")
                return False
            
            final_segments = [s for s in final_segments if s is not None]
            full_sound = call(final_segments, "Concatenate")
            
            # 最终强制对齐裁剪
            if full_sound.duration > target_total_duration:
                full_sound = full_sound.extract_part(0, target_total_duration, preserve_times=False)
            
            # 防止削波
            call(full_sound, "Scale peak", 0.99)
            
            full_sound.save(output_path, "WAV")
            
            if self.verbose:
                print(f"✅ 处理完成！(间隙已全部替换为静音)")
                print(f"目标时长: {target_total_duration:.3f}s | 结果时长: {full_sound.duration:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"Error during transformation: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    transformer = AudioTransformer(verbose=True)
    
    success = transformer.transform(
        source_audio_path="mel_convert/test/test_short1_1.wav",
        source_textgrid_path="mel_convert/test/aligned/test_short1_1.TextGrid",
        target_textgrid_path="mel_convert/test/aligned/test_short1_2.TextGrid",
        target_audio_path="mel_convert/test/test_short1_2.wav",
        output_path="output_natural.wav",
        tier_name="words"
    )
