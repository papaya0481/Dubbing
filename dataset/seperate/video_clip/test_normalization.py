#!/usr/bin/env python3
"""
测试文本正则化的时间戳对应性和缩写处理
"""

import sys
sys.path.insert(0, '/home/ruixin/workspace/dataset/seperate/video_clip')

from utils.text_normalization import normalize_text
from utils.subtitle_utils import normalize_sentence_info, Text2SRT
import json

def test_timestamp_preservation():
    """测试 1：验证时间戳不会因为正则化而改变"""
    print("\n" + "="*60)
    print("测试 1：验证时间戳保持不变")
    print("="*60)
    
    # 模拟一个 sentence_info 对象
    sentence_list = [
        {
            'text': '测试一二三四五',
            'timestamp': [[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
            'spk': 0
        },
        {
            'text': 'Hello world',
            'timestamp': [[2500, 3000], [3000, 3500]],
            'spk': 1
        }
    ]
    
    # 记录原始日期戳
    original_timestamps = []
    for sent in sentence_list:
        original_timestamps.append(sent['timestamp'].copy())
    
    print("\n原始句子：")
    for i, sent in enumerate(sentence_list):
        print(f"  {i}. text='{sent['text']}' timestamp={sent['timestamp']}")
    
    # 应用正则化
    print("\n应用正则化...")
    normalize_sentence_info(sentence_list)
    
    print("\n正则化后：")
    for i, sent in enumerate(sentence_list):
        print(f"  {i}. text='{sent['text']}' timestamp={sent['timestamp']}")
    
    # 验证时间戳没有改变
    print("\n时间戳验证：")
    all_ok = True
    for i, sent in enumerate(sentence_list):
        if sent['timestamp'] == original_timestamps[i]:
            print(f"  ✓ 句子 {i}：时间戳保持不变")
        else:
            print(f"  ✗ 句子 {i}：时间戳已改变！")
            print(f"    原始：{original_timestamps[i]}")
            print(f"    现在：{sent['timestamp']}")
            all_ok = False
    
    return all_ok


def test_abbreviation_skipping():
    """测试 2：验证英文缩写被正确跳过"""
    print("\n" + "="*60)
    print("测试 2：验证英文缩写被正确跳过")
    print("="*60)
    
    test_cases = [
        ("here's the answer", "here's", True),   # 应该被跳过
        ("dont worry", "dont worry", False),     # 不是缩写格式，可能被正则化
        ("it's great", "it's", True),            # 应该被跳过
        ("I'm happy", "I'm", True),              # 应该被跳过
        ("that's wonderful", "that's", True),    # 应该被跳过
        ("normal text", "normal text", False),   # 没有缩写
        ("22 hours", "22 hours", False),         # 数字+单词
    ]
    
    print("\n缩写检测测试：")
    for text, description, should_skip in test_cases:
        normalized = normalize_text(text)
        skipped = (text == normalized) and ("'" in text)
        
        status = "✓" if (skipped == should_skip) else "✗"
        print(f"  {status} '{description}': ", end="")
        if text == normalized:
            print(f"被跳过（保持原文本）")
        else:
            print(f"被正则化: '{text}' → '{normalized}'")


def test_text2srt_timestamp_calculation():
    """测试 3：验证 Text2SRT 的时间戳计算不依赖文本长度"""
    print("\n" + "="*60)
    print("测试 3：验证 Text2SRT 时间戳计算")
    print("="*60)
    
    # 创建两个句子，一个短，一个长，但时间戳相同
    timestamp = [[0, 1000], [1000, 2000], [2000, 3000]]
    
    text_short = "Hi"
    text_long = "Hello world this is a much longer text"
    
    srt_short = Text2SRT(text_short, timestamp, normalize=False)
    srt_long = Text2SRT(text_long, timestamp, normalize=False)
    
    print(f"\n短文本: '{text_short}'")
    print(f"  开始时间: {srt_short.start_time}")
    print(f"  结束时间: {srt_short.end_time}")
    
    print(f"\n长文本: '{text_long}'")
    print(f"  开始时间: {srt_long.start_time}")
    print(f"  结束时间: {srt_long.end_time}")
    
    if srt_short.start_time == srt_long.start_time and srt_short.end_time == srt_long.end_time:
        print("\n✓ 验证通过：时间戳与文本长度无关")
        return True
    else:
        print("\n✗ 验证失败：时间戳与文本长度有关")
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("文本正则化测试套件")
    print("="*60)
    
    results = []
    
    # 运行测试
    results.append(("时间戳保持不变", test_timestamp_preservation()))
    test_abbreviation_skipping()  # 缩写测试（直观检查）
    results.append(("Text2SRT 时间戳计算", test_text2srt_timestamp_calculation()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}：{name}")
    
    all_pass = all(success for _, success in results)
    if all_pass:
        print("\n✓ 所有关键测试通过")
        sys.exit(0)
    else:
        print("\n✗ 有测试失败")
        sys.exit(1)
