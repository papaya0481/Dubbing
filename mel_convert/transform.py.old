import parselmouth
from parselmouth.praat import call
import tgt

# 辅助函数：创建纯静音（仅作为 A 完全没有间隙时的保底方案）
def create_silence(duration, sample_rate):
    if duration <= 0.001: return None
    return call("Create Sound from formula", "silence", 1, 0, duration, sample_rate, "0")

# 辅助函数：提取音频
def extract_audio(sound, start, end):
    # 避免无效提取
    if end <= start: return None
    return sound.extract_part(from_time=start, to_time=end, preserve_times=False)

# 辅助函数：变速处理
def time_stretch(sound, stretch_factor):
    """
    智能时间拉伸：
    - 长片段：使用 PSOLA 算法（保持音质和音高）
    - 短片段：使用重采样（防止 Praat 报错）
    """
    if sound is None: return None
    
    # 限制拉伸倍率
    if stretch_factor < 0.1: stretch_factor = 0.1
    if stretch_factor > 8.0: stretch_factor = 8.0

    # === 分支 1：处理极短片段 (防止 Pitch Analysis 报错) ===
    # 阈值设为 0.05s (约 3 个 60Hz 周期)
    if sound.duration < 0.05:
        # 使用“磁带变速”原理：
        # 1. 修改采样率属性（欺骗播放速度，时长随之改变）
        #    stretch_factor > 1 (变慢) -> 降低采样率
        target_sr = sound.sampling_frequency / stretch_factor
        
        # 复制对象以免影响原数据
        temp_sound = sound.copy()
        temp_sound.override_sampling_frequency(target_sr)
        
        # 2. 重采样回原始采样率（为了能和其他片段拼接）
        #    这步操作会保持拉伸后的时长，但把采样率变回标准值
        resampled = temp_sound.resample(new_frequency=sound.sampling_frequency)
        return resampled

    # === 分支 2：处理正常片段 (PSOLA) ===
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
        # 如果万一 PSOLA 还是失败（比如音频全是静音无法检测音高），
        # 捕获异常并回退到上面的简单重采样方法
        # print(f"PSOLA failed ({e}), falling back to resampling...")
        target_sr = sound.sampling_frequency / stretch_factor
        temp_sound = sound.copy()
        temp_sound.override_sampling_frequency(target_sr)
        return temp_sound.resample(new_frequency=sound.sampling_frequency)

# ================= 主程序 =================

# 1. 加载资源
tg_a = tgt.io.read_textgrid("mel_convert/test/aligned/test_short1_1.TextGrid")
tg_b = tgt.io.read_textgrid("mel_convert/test/aligned/test_short1_2.TextGrid")
sound_a = parselmouth.Sound("mel_convert/test/test_short1_1.wav")

# 获取目标总时长用于最后校验
sound_b_ref = parselmouth.Sound("mel_convert/test/test_short1_2.wav")
target_total_duration = sound_b_ref.duration
original_sr = sound_a.sampling_frequency

print(f"Target Duration: {target_total_duration:.3f}s")

# 2. 提取实词列表 (过滤掉空白和sil)
# 我们假设 A 和 B 的实词内容和顺序是严格对应的
def get_real_words(tier):
    return [i for i in tier if i.text not in ['', 'sp', 'sil', '<eps>']]

tier_a = tg_a.get_tier_by_name("words")
tier_b = tg_b.get_tier_by_name("words")

words_a = get_real_words(tier_a)
words_b = get_real_words(tier_b)

if len(words_a) != len(words_b):
    print(f"Error: 单词数量不匹配! A:{len(words_a)} vs B:{len(words_b)}")
    # 在实际工程中这里可能需要更复杂的对齐逻辑，但在你的案例中应该是一样的
    exit()

final_segments = []

# 指针，记录上一个词结束的时间，用于计算 Gap
last_end_a = 0.0
last_end_b = 0.0

# 3. 循环遍历每一个单词
for i in range(len(words_b)):
    word_a = words_a[i]
    word_b = words_b[i]
    
    # -------------------------------------------------
    # A. 处理【词前间隙】 (Pre-Word Gap)
    # -------------------------------------------------
    # 目标间隙时长 (B)
    gap_duration_b = word_b.start_time - last_end_b
    
    # 源间隙时长 (A)
    gap_duration_a = word_a.start_time - last_end_a
    
    # 如果 B 需要间隙 (哪怕只有 0.01s)
    if gap_duration_b > 0.001:
        segment_to_append = None
        
        # 情况 1: A 在这里也有间隙（这就是你要的：提取 A 的底噪/呼吸）
        if gap_duration_a > 0.001:
            # 提取 A 的这段间隙
            raw_gap_a = extract_audio(sound_a, last_end_a, word_a.start_time)
            # 计算拉伸比：把 A 的 Gap 拉伸/压缩成 B 的 Gap
            stretch_ratio = gap_duration_b / gap_duration_a
            segment_to_append = time_stretch(raw_gap_a, stretch_ratio)
            # print(f"Gap填充: 使用 A({gap_duration_a:.2f}s) -> 拉伸至 B({gap_duration_b:.2f}s)")
            
        # 情况 2: B 有间隙，但 A 是连读（A 间隙为 0）
        else:
            # 这种情况下 A 没有底噪给我们用，只能生成静音
            # 或者你可以选择从 A 的开头截取一段底噪复用（这里先用生成静音）
            segment_to_append = create_silence(gap_duration_b, original_sr)
            # print(f"Gap填充: A 无间隙，生成静音 {gap_duration_b:.2f}s")
            
        if segment_to_append:
            final_segments.append(segment_to_append)
            
    # 如果 gap_duration_b <= 0，说明 B 是紧密连读的，我们直接跳过 A 的这段间隙（Discard A's gap）
    
    # -------------------------------------------------
    # B. 处理【单词本身】 (Word)
    # -------------------------------------------------
    duration_target = word_b.end_time - word_b.start_time
    duration_source = word_a.end_time - word_a.start_time
    
    raw_word_a = extract_audio(sound_a, word_a.start_time, word_a.end_time)
    
    if duration_target > 0.001 and raw_word_a:
        stretch_ratio = duration_target / duration_source
        processed_word = time_stretch(raw_word_a, stretch_ratio)
        final_segments.append(processed_word)
    
    # 更新指针
    last_end_a = word_a.end_time
    last_end_b = word_b.end_time

# 4. 处理【尾部间隙】 (Tail Gap)
# 最后一个词结束到音频 B 结束
tail_gap_b = target_total_duration - last_end_b
tail_gap_a = sound_a.duration - last_end_a # A 的剩余时长

if tail_gap_b > 0.001:
    if tail_gap_a > 0.001:
        # 使用 A 的尾部底噪
        raw_tail_a = extract_audio(sound_a, last_end_a, sound_a.duration)
        stretch_ratio = tail_gap_b / tail_gap_a
        processed_tail = time_stretch(raw_tail_a, stretch_ratio)
        final_segments.append(processed_tail)
    else:
        # A 没了，补静音
        final_segments.append(create_silence(tail_gap_b, original_sr))

# 5. 拼接与保存
if final_segments:
    # 过滤 None
    final_segments = [s for s in final_segments if s is not None]
    full_sound = call(final_segments, "Concatenate")
    
    # 最终强制对齐裁剪 (消除浮点误差)
    if full_sound.duration > target_total_duration:
        full_sound = full_sound.extract_part(0, target_total_duration, preserve_times=False)
        
    full_sound.save("output_natural.wav", "WAV")
    print(f"✅ 处理完成！保留了 A 的环境信息。")
    print(f"目标时长: {target_total_duration:.3f}s | 结果时长: {full_sound.duration:.3f}s")
else:
    print("Error: 未生成音频")