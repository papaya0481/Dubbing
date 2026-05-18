#!/usr/bin/env python3
"""
文本正则化模块，使用 WeTextProcessing 库
"""

from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


class TextNormalizer:
    """文本正则化类，支持中英文"""
    
    _instance = None
    _tn_en = None
    _tn_zh = None
    
    def __new__(cls):
        """单例模式，确保只初始化一次"""
        if cls._instance is None:
            cls._instance = super(TextNormalizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化文本正则化器"""
        if self._initialized:
            return
        
        try:
            from tn.chinese.normalizer import Normalizer as ChineseNormalizer
            from tn.english.normalizer import Normalizer as EnglishNormalizer
            
            self._tn_zh = ChineseNormalizer()
            self._tn_en = EnglishNormalizer()
            logger.info("✓ WeTextProcessing 文本正则化器初始化成功")
        except ImportError as e:
            logger.warning(f"⚠ 无法导入 WeTextProcessing: {e}")
            logger.warning("  请运行: pip install WeTextProcessing")
            self._tn_zh = None
            self._tn_en = None
        except Exception as e:
            logger.warning(f"⚠ 文本正则化器初始化失败: {e}")
            self._tn_zh = None
            self._tn_en = None
        
        self._initialized = True
    
    def _detect_language(self, text: str) -> str:
        """
        检测文本主要语言
        
        Args:
            text: 输入文本
            
        Returns:
            'zh' 或 'en'
        """
        if not text:
            return 'en'
        
        # 统计中文字符比例
        zh_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)
        
        if total_chars == 0:
            return 'en'
        
        # 如果中文字符超过 30%，判定为中文
        return 'zh' if zh_count / total_chars > 0.3 else 'en'
    
    def normalize(self, text: str, language: Optional[str] = None) -> str:
        """
        对文本进行正则化处理
        
        Args:
            text: 输入文本
            language: 语言类型 ('zh', 'en', 或 None 自动检测)
            
        Returns:
            正则化后的文本
        """
        if not text or not isinstance(text, str):
            return text
        
        text = text.strip()
        if not text:
            return text
        
        # 检测英文缩写（如 here's, don't, it's 等）
        # 模式：单词 + 撇号 + 小写字母（常见英文缩写）
        if re.search(r"\w+'[a-z]", text, re.IGNORECASE):
            logger.debug(f"检测到缩写，跳过正则化: {text}")
            return text
        
        # 如果无法初始化 WeTextProcessing，返回原文本
        if self._tn_zh is None and self._tn_en is None:
            return text
        
        try:
            # 自动检测语言
            if language is None:
                language = self._detect_language(text)
            
            # 根据语言选择正则化器
            if language == 'zh' and self._tn_zh is not None:
                normalized = self._tn_zh.normalize(text)
            elif language == 'en' and self._tn_en is not None:
                normalized = self._tn_en.normalize(text)
            else:
                # 如果没有对应的正则化器，返回原文本
                normalized = text
            
            return normalized if normalized else text
            
        except Exception as e:
            logger.warning(f"文本正则化失败: {e}，返回原文本")
            return text
    
    def normalize_batch(self, texts: list, language: Optional[str] = None) -> list:
        """
        批量正则化文本
        
        Args:
            texts: 文本列表
            language: 语言类型
            
        Returns:
            正则化后的文本列表
        """
        return [self.normalize(text, language) for text in texts]


# 全局实例
_normalizer = None


def get_normalizer() -> TextNormalizer:
    """获取全局文本正则化器实例"""
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer


def normalize_text(text: str, language: Optional[str] = None) -> str:
    """
    全局文本正则化函数
    
    Args:
        text: 输入文本
        language: 语言类型 ('zh', 'en', 或 None 自动检测)
        
    Returns:
        正则化后的文本
        
    Example:
        >>> normalized = normalize_text("我来自中国。")
        >>> print(normalized)
    """
    normalizer = get_normalizer()
    return normalizer.normalize(text, language)


def normalize_text_batch(texts: list, language: Optional[str] = None) -> list:
    """
    全局批量文本正则化函数
    
    Args:
        texts: 文本列表
        language: 语言类型
        
    Returns:
        正则化后的文本列表
    """
    normalizer = get_normalizer()
    return normalizer.normalize_batch(texts, language)


if __name__ == "__main__":
    # 测试代码
    test_cases = [
        "我来自中国。",
        "The quick brown fox jumps over the lazy dog.",
        "Hello3world，你好，测试，一二三。",
        "123，456.789 RMB",
    ]
    
    print("=" * 60)
    print("文本正则化测试")
    print("=" * 60)
    
    normalizer = get_normalizer()
    for text in test_cases:
        normalized = normalizer.normalize(text)
        print(f"原文本: {text}")
        print(f"正规化: {normalized}")
        print()
