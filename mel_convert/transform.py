"""
音频变换器：根据目标TextGrid的时间轴，将源音频进行变速和间隙调整
支持两种后端：
1. Praat (PSOLA/Resample) - 使用 parselmouth
2. Librosa (Phase Vocoder) - 使用 librosa + numpy
"""

import tgt
from pathlib import Path
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod


class BaseAudioTransformer(ABC):
    """音频变换器基类"""
    
    def __init__(self, 
                 min_gap_threshold: float = 0.001, 
                 max_stretch: float = 8.0,
                 verbose: bool = True):
        """
        Args:
            min_gap_threshold: 最小间隙阈值（秒）
            max_stretch: 最大拉伸倍率
            verbose: 是否输出详细信息
        """
        self.min_gap_threshold = min_gap_threshold
        self.max_stretch = max_stretch
        self.verbose = verbose
    
    @abstractmethod
    def load_audio(self, path: str):
        """加载音频文件"""
        pass
    
    @abstractmethod
    def get_duration(self, audio) -> float:
        """获取音频时长"""
        pass
    
    @abstractmethod
    def get_sample_rate(self, audio) -> int:
        """获取采样率"""
        pass
    
    @abstractmethod
    def extract_segment(self, audio, start: float, end: float):
        """提取音频片段"""
        pass
    
    @abstractmethod
    def create_silence(self, duration: float, sample_rate: int):
        """创建静音片段"""
        pass
    
    @abstractmethod
    def time_stretch(self, audio_segment, stretch_factor: float, is_noise: bool = False):
        """时间拉伸"""
        pass
    
    @abstractmethod
    def concatenate_segments(self, segments: List, sample_rate: int):
        """拼接音频片段"""
        pass
    
    @abstractmethod
    def save_audio(self, audio, output_path: str):
        """保存音频"""
        pass
    
    def get_real_words(self, tier: tgt.IntervalTier) -> List[tgt.Interval]:
        """提取实词列表（过滤掉空白和静音）"""
        return [i for i in tier if i.text not in ['', 'sp', 'sil', '<eps>']]
    
    def transform(self, 
                  source_audio_path: str,
                  source_textgrid_path: str,
                  target_textgrid_path: str,
                  target_audio_path: str,
                  output_path: str,
                  tier_name: str = "words") -> bool:
        """
        执行音频变换
        
        Args:
            source_audio_path: 源音频文件路径
            source_textgrid_path: 源TextGrid文件路径
            target_textgrid_path: 目标TextGrid文件路径
            target_audio_path: 目标音频文件路径（用于获取目标时长）
            output_path: 输出音频文件路径
            tier_name: TextGrid中的层级名称
            
        Returns:
            是否成功
        """
        try:
            if self.verbose:
                print(f"Loading resources with {self.__class__.__name__}...")
            
            # 1. 加载资源
            audio_source = self.load_audio(source_audio_path)
            audio_target_ref = self.load_audio(target_audio_path)
            
            target_total_duration = self.get_duration(audio_target_ref)
            sample_rate = self.get_sample_rate(audio_source)
            
            tg_source = tgt.io.read_textgrid(source_textgrid_path)
            tg_target = tgt.io.read_textgrid(target_textgrid_path)
            
            if self.verbose:
                print(f"Source duration: {self.get_duration(audio_source):.3f}s")
                print(f"Target duration: {target_total_duration:.3f}s")
            
            # 2. 提取实词列表
            tier_source = tg_source.get_tier_by_name(tier_name)
            tier_target = tg_target.get_tier_by_name(tier_name)
            
            words_source = self.get_real_words(tier_source)
            words_target = self.get_real_words(tier_target)
            
            if len(words_source) != len(words_target):
                print(f"Error: 单词数量不匹配! Source:{len(words_source)} vs Target:{len(words_target)}")
                return False
            
            if self.verbose:
                print(f"Processing {len(words_target)} words...")
            
            final_segments = []
            last_end_source = 0.0
            last_end_target = 0.0
            
            # 3. 处理每个单词
            for i in range(len(words_target)):
                word_source = words_source[i]
                word_target = words_target[i]
                
                # A. 处理词前间隙
                gap_duration_target = word_target.start_time - last_end_target
                gap_duration_source = word_source.start_time - last_end_source
                
                if gap_duration_target > self.min_gap_threshold:
                    segment_to_append = None
                    
                    if gap_duration_source > self.min_gap_threshold:
                        # 使用源音频的间隙（保留底噪/呼吸）
                        raw_gap = self.extract_segment(audio_source, last_end_source, word_source.start_time)
                        if raw_gap is not None:
                            stretch_ratio = gap_duration_target / gap_duration_source
                            segment_to_append = self.time_stretch(raw_gap, stretch_ratio, is_noise=True)
                    else:
                        # 源音频无间隙，生成静音
                        segment_to_append = self.create_silence(gap_duration_target, sample_rate)
                    
                    if segment_to_append is not None:
                        final_segments.append(segment_to_append)
                
                # B. 处理单词本身
                duration_target = word_target.end_time - word_target.start_time
                duration_source = word_source.end_time - word_source.start_time
                
                raw_word = self.extract_segment(audio_source, word_source.start_time, word_source.end_time)
                
                if duration_target > self.min_gap_threshold and raw_word is not None:
                    stretch_ratio = duration_target / duration_source
                    processed_word = self.time_stretch(raw_word, stretch_ratio, is_noise=False)
                    if processed_word is not None:
                        final_segments.append(processed_word)
                
                # 更新指针
                last_end_source = word_source.end_time
                last_end_target = word_target.end_time
            
            # 4. 处理尾部间隙
            tail_gap_target = target_total_duration - last_end_target
            tail_gap_source = self.get_duration(audio_source) - last_end_source
            
            if tail_gap_target > self.min_gap_threshold:
                if tail_gap_source > self.min_gap_threshold:
                    raw_tail = self.extract_segment(audio_source, last_end_source, self.get_duration(audio_source))
                    if raw_tail is not None:
                        stretch_ratio = tail_gap_target / tail_gap_source
                        processed_tail = self.time_stretch(raw_tail, stretch_ratio, is_noise=True)
                        if processed_tail is not None:
                            final_segments.append(processed_tail)
                else:
                    tail_silence = self.create_silence(tail_gap_target, sample_rate)
                    if tail_silence is not None:
                        final_segments.append(tail_silence)
            
            # 5. 拼接与保存
            if not final_segments:
                print("Error: 未生成音频片段")
                return False
            
            if self.verbose:
                print(f"Concatenating {len(final_segments)} segments...")
            
            full_audio = self.concatenate_segments(final_segments, sample_rate)
            self.save_audio(full_audio, output_path)
            
            if self.verbose:
                print(f"✅ 处理完成！已保存到: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error during transformation: {e}")
            import traceback
            traceback.print_exc()
            return False


class PraatTransformer(BaseAudioTransformer):
    """基于 Parselmouth/Praat 的变换器"""
    
    def __init__(self,
                 min_gap_threshold: float = 0.001,
                 max_stretch: float = 8.0,
                 min_stretch: float = 0.1,
                 short_audio_threshold: float = 0.05,
                 pitch_range: Tuple[int, int] = (75, 600),
                 stretch_method: str = "psola",
                 verbose: bool = True):
        """
        Args:
            stretch_method: "psola" 或 "resample"
            pitch_range: (min_pitch, max_pitch) for PSOLA
            short_audio_threshold: 短音频阈值，低于此值自动使用resample
        """
        super().__init__(min_gap_threshold, max_stretch, verbose)
        self.min_stretch = min_stretch
        self.short_audio_threshold = short_audio_threshold
        self.pitch_range = pitch_range
        self.stretch_method = stretch_method.lower()
        
        if self.stretch_method not in ["psola", "resample"]:
            raise ValueError(f"stretch_method must be 'psola' or 'resample', got '{stretch_method}'")
        
        # 延迟导入
        import parselmouth
        from parselmouth.praat import call
        self.parselmouth = parselmouth
        self.call = call
    
    def load_audio(self, path: str):
        return self.parselmouth.Sound(path)
    
    def get_duration(self, audio) -> float:
        return audio.duration
    
    def get_sample_rate(self, audio) -> int:
        return audio.sampling_frequency
    
    def extract_segment(self, audio, start: float, end: float):
        if end <= start:
            return None
        return audio.extract_part(from_time=start, to_time=end, preserve_times=False)
    
    def create_silence(self, duration: float, sample_rate: int):
        if duration <= self.min_gap_threshold:
            return None
        return self.call("Create Sound from formula", "silence", 1, 0, duration, sample_rate, "0")
    
    def time_stretch(self, audio_segment, stretch_factor: float, is_noise: bool = False):
        if audio_segment is None:
            return None
        
        stretch_factor = max(self.min_stretch, min(stretch_factor, self.max_stretch))
        original_sr = audio_segment.sampling_frequency
        
        # 使用 Resample 方法
        if self.stretch_method == "resample" or is_noise:
            target_sr = original_sr / stretch_factor
            temp_sound = audio_segment.copy()
            temp_sound.override_sampling_frequency(target_sr)
            return temp_sound.resample(new_frequency=original_sr)
        
        # 使用 PSOLA 方法
        if audio_segment.duration < self.short_audio_threshold:
            if self.verbose:
                print(f"Warning: Audio too short ({audio_segment.duration:.3f}s), using resample")
            target_sr = original_sr / stretch_factor
            temp_sound = audio_segment.copy()
            temp_sound.override_sampling_frequency(target_sr)
            return temp_sound.resample(new_frequency=original_sr)
        
        try:
            manipulated = self.call(audio_segment, "To Manipulation", 0.01, 
                                   self.pitch_range[0], self.pitch_range[1])
            duration_tier = self.call(manipulated, "Extract duration tier")
            
            end_time = self.call(duration_tier, "Get end time")
            self.call(duration_tier, "Remove points between", 0.0, end_time)
            self.call(duration_tier, "Add point", 0.0, stretch_factor)
            
            self.call([manipulated, duration_tier], "Replace duration tier")
            stretched = self.call(manipulated, "Get resynthesis (overlap-add)")
            return stretched
            
        except Exception as e:
            if self.verbose:
                print(f"PSOLA failed ({e}), falling back to resample")
            target_sr = original_sr / stretch_factor
            temp_sound = audio_segment.copy()
            temp_sound.override_sampling_frequency(target_sr)
            return temp_sound.resample(new_frequency=original_sr)
    
    def concatenate_segments(self, segments: List, sample_rate: int):
        segments = [s for s in segments if s is not None]
        if not segments:
            return None
        return self.call(segments, "Concatenate")
    
    def save_audio(self, audio, output_path: str):
        # 防止削波
        self.call(audio, "Scale peak", 0.99)
        audio.save(output_path, "WAV")


class LibrosaTransformer(BaseAudioTransformer):
    """基于 Librosa + NumPy 的变换器"""
    
    def __init__(self,
                 min_gap_threshold: float = 0.001,
                 max_stretch: float = 8.0,
                 crossfade_ms: float = 5.0,
                 verbose: bool = True):
        """
        Args:
            crossfade_ms: 拼接处的淡入淡出时长(毫秒)
        """
        super().__init__(min_gap_threshold, max_stretch, verbose)
        self.crossfade_duration = crossfade_ms / 1000.0
        
        # 延迟导入
        import librosa
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample
        
        self.librosa = librosa
        self.np = np
        self.sf = sf
        self.resample = resample
    
    def load_audio(self, path: str):
        """返回 (y, sr) 元组"""
        y, sr = self.librosa.load(path, sr=None, mono=True)
        return (y, sr)
    
    def get_duration(self, audio) -> float:
        y, sr = audio
        return len(y) / sr
    
    def get_sample_rate(self, audio) -> int:
        y, sr = audio
        return sr
    
    def extract_segment(self, audio, start: float, end: float):
        y, sr = audio
        if end <= start:
            return None
        
        start_idx = int(self.np.round(start * sr))
        end_idx = int(self.np.round(end * sr))
        
        start_idx = max(0, start_idx)
        end_idx = min(len(y), end_idx)
        
        if start_idx >= end_idx:
            return None
        
        return y[start_idx:end_idx]
    
    def create_silence(self, duration: float, sample_rate: int):
        if duration <= self.min_gap_threshold:
            return None
        num_samples = int(duration * sample_rate)
        return self.np.zeros(num_samples, dtype=self.np.float32)
    
    def time_stretch(self, audio_segment, stretch_factor: float, is_noise: bool = False):
        if audio_segment is None or len(audio_segment) == 0:
            return None
        
        stretch_factor = max(0.1, min(stretch_factor, self.max_stretch))
        target_len = int(len(audio_segment) * stretch_factor)
        
        if target_len < 1:
            return audio_segment
        
        # 对于底噪或极短片段，使用线性重采样
        if is_noise or len(audio_segment) < 2048:
            return self.resample(audio_segment, target_len)
        
        # 对于正常语音，使用 Phase Vocoder
        try:
            rate = 1.0 / stretch_factor
            y_stretched = self.librosa.effects.time_stretch(audio_segment, rate=rate)
            
            # 强制对齐长度
            if len(y_stretched) != target_len:
                y_stretched = self.resample(y_stretched, target_len)
            
            return y_stretched
            
        except Exception as e:
            if self.verbose:
                print(f"Phase Vocoder failed: {e}, falling back to resample")
            return self.resample(audio_segment, target_len)
    
    def concatenate_segments(self, segments: List, sample_rate: int):
        """带交叉淡入淡出的拼接"""
        segments = [s for s in segments if s is not None and len(s) > 0]
        if not segments:
            return None
        
        if len(segments) == 1:
            return segments[0]
        
        fade_samples = int(self.crossfade_duration * sample_rate)
        
        if fade_samples == 0:
            return self.np.concatenate(segments)
        
        result = segments[0]
        
        for i in range(1, len(segments)):
            next_seg = segments[i]
            
            if len(result) < fade_samples or len(next_seg) < fade_samples:
                result = self.np.concatenate((result, next_seg))
                continue
            
            # Crossfade
            overlap_a = result[-fade_samples:]
            pre_a = result[:-fade_samples]
            
            overlap_b = next_seg[:fade_samples]
            post_b = next_seg[fade_samples:]
            
            fade_in = self.np.linspace(0, 1, fade_samples)
            fade_out = 1.0 - fade_in
            
            merged_overlap = (overlap_a * fade_out) + (overlap_b * fade_in)
            result = self.np.concatenate((pre_a, merged_overlap, post_b))
        
        return result
    
    def save_audio(self, audio, output_path: str):
        # 峰值归一化
        max_val = self.np.max(self.np.abs(audio))
        if max_val > 0.99:
            audio = audio * (0.99 / max_val)
        
        # 从缓存的 audio 中获取采样率
        # 注意：此时 audio 已经是 numpy array，需要从原始音频获取 sr
        # 这里我们需要存储 sr，所以修改 concatenate_segments 返回
        # 简单起见，我们在 transform 中传递 sr
        # 但为了保持接口一致，我们需要改进设计
        
        # 临时方案：假设我们有 self.current_sr
        if hasattr(self, '_current_sr'):
            sr = self._current_sr
        else:
            sr = 22050  # 默认值
        
        self.sf.write(output_path, audio, sr)
    
    def transform(self, *args, **kwargs):
        """重写 transform 以保存采样率"""
        # 提取源音频以获取采样率
        source_audio_path = args[0] if args else kwargs.get('source_audio_path')
        if source_audio_path:
            temp_audio = self.load_audio(source_audio_path)
            self._current_sr = self.get_sample_rate(temp_audio)
        
        return super().transform(*args, **kwargs)


def create_transformer(method: str = "librosa", **kwargs) -> BaseAudioTransformer:
    """
    工厂函数：创建音频变换器
    
    Args:
        method: "praat" 或 "librosa"
        **kwargs: 传递给具体变换器的参数
        
    Returns:
        BaseAudioTransformer 实例
    """
    method = method.lower()
    
    if method == "praat":
        return PraatTransformer(**kwargs)
    elif method == "librosa":
        return LibrosaTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'praat' or 'librosa'")


if __name__ == "__main__":
    # 示例 1: 使用 Praat + PSOLA
    print("=" * 60)
    print("示例 1: Praat + PSOLA")
    print("=" * 60)
    transformer1 = create_transformer(
        method="praat",
        stretch_method="psola",
        verbose=True
    )
    
    transformer1.transform(
        source_audio_path="mel_convert/test/test_short1_1.wav",
        source_textgrid_path="mel_convert/test/aligned/test_short1_1.TextGrid",
        target_textgrid_path="mel_convert/test/aligned/test_short1_2.TextGrid",
        target_audio_path="mel_convert/test/test_short1_2.wav",
        output_path="output_praat_psola.wav",
        tier_name="words"
    )
    
    print("\n" + "=" * 60)
    print("示例 2: Librosa + Phase Vocoder")
    print("=" * 60)
    
    # 示例 2: 使用 Librosa + Phase Vocoder
    transformer2 = create_transformer(
        method="librosa",
        crossfade_ms=5.0,
        verbose=True
    )
    
    transformer2.transform(
        source_audio_path="mel_convert/test/test_short1_1.wav",
        source_textgrid_path="mel_convert/test/aligned/test_short1_1.TextGrid",
        target_textgrid_path="mel_convert/test/aligned/test_short1_2.TextGrid",
        target_audio_path="mel_convert/test/test_short1_2.wav",
        output_path="output_librosa.wav",
        tier_name="words"
    )
