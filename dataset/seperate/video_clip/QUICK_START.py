#!/usr/bin/env python3
"""
快速参考指南 - WeTextProcessing 文本正规化

快速开始指南，包含常用代码示例
"""

# ============================================================================
# 1. 安装
# ============================================================================
"""
pip install WeTextProcessing
"""


# ============================================================================
# 2. 基础使用
# ============================================================================

# 方式 A: 全局函数（推荐简单场景）
from utils.text_normalization import normalize_text

result = normalize_text("我来自北京3号楼")
print(result)  # 我来自北京三号楼


# 方式 B: 单例实例（推荐复杂场景）
from utils.text_normalization import get_normalizer

normalizer = get_normalizer()
result = normalizer.normalize("The price is $99.99")
print(result)  # The price is ninety nine point nine nine dollars


# 方式 C: 批量处理
texts = ["文本1", "文本2", "文本3"]
results = normalizer.normalize_batch(texts)


# ============================================================================
# 3. 在 SRT 生成中使用
# ============================================================================

from utils.subtitle_utils import Text2SRT, generate_srt, generate_srt_clip

# 单个 Text2SRT 对象
t2s = Text2SRT(
    text="hello world",
    timestamp=[[100, 500], [500, 1000]],
    offset=0,
    normalize=True  # 启用正规化
)
print(t2s.text())  # "hello world."


# generate_srt 函数
sentence_list = [
    {'text': '我来自中国', 'timestamp': [[100, 500], [500, 1000], [1000, 1500]]},
    {'text': '北京市', 'timestamp': [[1500, 2000], [2000, 2500]]},
]
srt = generate_srt(sentence_list, normalize=True)
print(srt)


# generate_srt_clip 函数
srt_clip, subs, index = generate_srt_clip(
    sentence_list,
    start=0.1,
    end=2.5,
    normalize=True  # 启用正规化
)


# ============================================================================
# 4. 在 videoclipper_v2.py 中使用
# ============================================================================

"""
# 位置 1 (约第354行)
res_srt = generate_srt(sentence_info, normalize=True)

# 位置 2 (约第384行)
res_srt = generate_srt(rec_result[0]['sentence_info'], normalize=True)

# 位置 3 (约第445行)
srt_clip, subs, srt_index = generate_srt_clip(
    sentence_list,
    start,
    end,
    begin_index=begin_index,
    time_acc_ost=time_acc_ost,
    normalize=True  # 添加此参数
)
"""


# ============================================================================
# 5. 指定语言（高级用法）
# ============================================================================

# 强制中文处理
result_zh = normalize_text("Hello 世界", language='zh')

# 强制英文处理
result = en = normalize_text("2024年", language='en')

# 自动检测（默认）
result = normalize_text("Hello 世界")  # 自动判断


# ============================================================================
# 6. 禁用正规化
# ============================================================================

# 某些情况需要禁用正规化，保留原文本
t2s_no_norm = Text2SRT(
    text="原始文本",
    timestamp=[[100, 500]],
    normalize=False  # 禁用正规化
)

srt_no_norm = generate_srt(sentence_list, normalize=False)


# ============================================================================
# 7. 调试和日志
# ============================================================================

import logging

# 启用调试日志
logging.basicConfig(level=logging.DEBUG)

# 现在会看到详细的初始化和处理信息
from utils.text_normalization import get_normalizer
normalizer = get_normalizer()


# ============================================================================
# 8. 保存结果到文件
# ============================================================================

# 保存正规化 SRT
srt = generate_srt(sentence_list, normalize=True)
with open('output.srt', 'w', encoding='utf-8') as f:
    f.write(srt)

# 保存正规化文本
text = normalize_text("原始文本")
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(text)


# ============================================================================
# 9. 常见场景
# ============================================================================

# 场景 A: ASR 识别后直接正规化
def process_asr_result(asr_text):
    normalized = normalize_text(asr_text)
    return normalized

# 场景 B: 生成字幕前正规化
def create_subtitles(sentence_info, output_file):
    srt = generate_srt(sentence_info, normalize=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt)

# 场景 C: 处理混合语言文本
def process_mixed_text(text):
    normalizer = get_normalizer()
    return normalizer.normalize(text)  # 自动检测语言


# ============================================================================
# 10. 性能提示
# ============================================================================

"""
✅ 高效做法:
- 使用单例模式 (get_normalizer()) 避免重复初始化
- 对大量文本使用 normalize_batch()
- 缓存常用文本的正规化结果

❌ 低效做法:
- 每次都创建新的 TextNormalizer 实例
- 逐个处理大量文本而不用 batch
- 重复正规化相同文本
"""


# ============================================================================
# 11. 错误处理
# ============================================================================

try:
    result = normalize_text("文本")
except Exception as e:
    print(f"正规化失败: {e}")
    # 会自动返回原文本，不中断流程

    
# ============================================================================
# 12. 完整工作流示例
# ============================================================================

def complete_workflow(asr_result):
    """完整的工作流示例"""
    
    # 1. 导入模块
    from utils.subtitle_utils import generate_srt, process_asr_to_sentence_info
    from utils.text_normalization import normalize_text
    
    # 2. 处理 ASR 结果
    sentence_info, full_text = process_asr_to_sentence_info(asr_result)
    
    # 3. 正规化 ASR 文本
    normalized_full_text = normalize_text(full_text)
    
    # 4. 生成 SRT（自动正规化）
    srt_content = generate_srt(sentence_info, normalize=True)
    
    # 5. 保存结果
    with open('subtitles.srt', 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    print(f"✓ 处理完成")
    print(f"  - 原文本: {full_text[:50]}...")
    print(f"  - 正规化: {normalized_full_text[:50]}...")
    print(f"  - SRT 行数: {srt_content.count(chr(10))}")


# ============================================================================
# 13. 测试
# ============================================================================

if __name__ == "__main__":
    # 运行快速测试
    print("快速测试:")
    print("-" * 50)
    
    test_cases = [
        "我的号码是12345",
        "The price is $99.99 USD",
        "2024年3月18日",
    ]
    
    for text in test_cases:
        normalized = normalize_text(text)
        print(f"原文: {text}")
        print(f"正规化: {normalized}")
        print()
