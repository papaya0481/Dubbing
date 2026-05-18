#!/usr/bin/env python3
"""
演示脚本：展示如何使用 WeTextProcessing 进行文本正则化
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.text_normalization import normalize_text, get_normalizer, TextNormalizer
from utils.subtitle_utils import Text2SRT, generate_srt


def demo_basic_normalization():
    """演示基础文本正则化"""
    print("\n" + "=" * 70)
    print("演示 1: 基础文本正则化")
    print("=" * 70)
    
    test_texts = [
        "我来自中国，北京市。",
        "The quick brown fox，jumps over the lazy dog.",
        "今天天气很好，我们去公园玩吧！",
        "2024年3月18日，星期一。",
        "价格是$100 USD，约¥700人民币。",
    ]
    
    normalizer = get_normalizer()
    
    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"\n原文本:  {text}")
        print(f"正规化:  {normalized}")


def demo_text2srt():
    """演示 Text2SRT 中的文本正则化"""
    print("\n" + "=" * 70)
    print("演示 2: Text2SRT 中的文本正则化")
    print("=" * 70)
    
    # 模拟时间戳数据 (以毫秒为单位)
    timestamps = [
        [100, 500],   # 第一个词
        [500, 1000],  # 第二个词
        [1000, 1500], # 第三个词
    ]
    
    # 测试中文文本
    print("\n--- 测试中文文本 ---")
    text_zh = "我来自中国"
    t2s_zh_with_norm = Text2SRT(text_zh, timestamps, normalize=True)
    t2s_zh_without_norm = Text2SRT(text_zh, timestamps, normalize=False)
    
    print(f"原文本:        {text_zh}")
    print(f"启用正规化:    {t2s_zh_with_norm.text()}")
    print(f"禁用正规化:    {t2s_zh_without_norm.text()}")
    
    # 测试英文文本
    print("\n--- 测试英文文本 ---")
    text_en = "Hello world"
    t2s_en_with_norm = Text2SRT(text_en, timestamps, normalize=True)
    t2s_en_without_norm = Text2SRT(text_en, timestamps, normalize=False)
    
    print(f"原文本:        {text_en}")
    print(f"启用正规化:    {t2s_en_with_norm.text()}")
    print(f"禁用正규化:    {t2s_en_without_norm.text()}")


def demo_generate_srt():
    """演示 generate_srt 中的文本正则化"""
    print("\n" + "=" * 70)
    print("演示 3: generate_srt 中的文本正则化")
    print("=" * 70)
    
    # 模拟 ASR 识别结果
    sentence_list = [
        {
            'text': '我来自中国',
            'timestamp': [[100, 500], [500, 1000], [1000, 1500]],
        },
        {
            'text': '北京市',
            'timestamp': [[1500, 2000], [2000, 2500]],
        },
    ]
    
    print("\n生成 SRT 格式文本（启用正规化）:")
    print("-" * 70)
    srt_with_norm = generate_srt(sentence_list, normalize=True)
    print(srt_with_norm)
    
    print("\n生成 SRT 格式文本（禁用正规化）:")
    print("-" * 70)
    srt_without_norm = generate_srt(sentence_list, normalize=False)
    print(srt_without_norm)


def demo_batch_normalization():
    """演示批量文本正则化"""
    print("\n" + "=" * 70)
    print("演示 4: 批量文本正则化")
    print("=" * 70)
    
    texts = [
        "我是来自深圳的一名工程师，",
        "今年 2024 年，已经是第 18 个年头了。",
        "The price is $99.99 for basic plan.",
    ]
    
    normalizer = get_normalizer()
    normalized_texts = normalizer.normalize_batch(texts)
    
    for orig, norm in zip(texts, normalized_texts):
        print(f"\n原文本: {orig}")
        print(f"正规化: {norm}")


def demo_video_processing():
    """演示在视频处理流程中的使用"""
    print("\n" + "=" * 70)
    print("演示 5: 在视频处理流程中的使用")
    print("=" * 70)
    
    print("""
在 videoclipper_v2.py 中的使用方式：

1. 在调用 generate_srt 时，设置 normalize=True：
   res_srt = generate_srt(sentence_info, normalize=True)

2. 在调用 generate_srt_clip 时，设置 normalize=True：
   srt_clip, subs, srt_index = generate_srt_clip(
       sentence_list, 
       start, 
       end, 
       normalize=True
   )

3. 创建 Text2SRT 对象时，设置 normalize=True：
   t2s = Text2SRT(sent['text'], sent['timestamp'], normalize=True)

文本流程：
   ASR识别结果 
        ↓
   process_asr_to_sentence_info()
        ↓
   生成 sentence_list
        ↓
   generate_srt(sentence_list, normalize=True)  ← 启用正规化
        ↓
   正规化后的 SRT 文本
        ↓
   保存到文件
    """)


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WeTextProcessing 文本正规化演示" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        demo_basic_normalization()
    except Exception as e:
        print(f"演示 1 出错: {e}")
    
    try:
        demo_text2srt()
    except Exception as e:
        print(f"演示 2 出错: {e}")
    
    try:
        demo_generate_srt()
    except Exception as e:
        print(f"演示 3 出错: {e}")
    
    try:
        demo_batch_normalization()
    except Exception as e:
        print(f"演示 4 出错: {e}")
    
    try:
        demo_video_processing()
    except Exception as e:
        print(f"演示 5 出错: {e}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
