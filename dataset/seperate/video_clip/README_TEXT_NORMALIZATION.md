# WeTextProcessing 文本正规化集成指南

## 概述

已将 **WeTextProcessing** 库集成到视频处理管道中，用于对识别到的文本进行正规化处理。文本经过正规化后会被保存到 SRT 字幕文件和元数据中。

## 功能特性

✅ **自动语言检测** - 自动识别中英文文本
✅ **中英文支持** - 分别处理中文和英文文本正规化
✅ **单例模式** - 确保模型只被加载一次，提高效率
✅ **灵活控制** - 可随时启用/禁用文本正规化
✅ **错误处理** - 异常捕获，确保不会中断流程

## 安装

### 1. 安装 WeTextProcessing

```bash
pip install WeTextProcessing
```

### 2. 验证安装

```bash
python -c "from tn.chinese import TextNormalizer; print('✓ 安装成功')"
```

## 架构

### 文件结构

```
seperate/video_clip/
├── utils/
│   ├── text_normalization.py    # 新增：文本正规化模块
│   ├── subtitle_utils.py        # 修改：集成正规化功能
│   └── ...
├── videoclipper_v2.py           # 使用文本正规化
├── demo_text_normalization.py   # 演示脚本
└── README_TEXT_NORMALIZATION.md # 本文件
```

### 核心模块

#### `text_normalization.py`

提供全局的文本正规化接口：

```python
from utils.text_normalization import normalize_text, get_normalizer

# 方式 1：直接调用全局函数
normalized = normalize_text("我的数字是123")

# 方式 2：获取单例实例
normalizer = get_normalizer()
normalized = normalizer.normalize("Hello 世界")

# 方式 3：批量处理
texts = ["文本1", "文本2"]
normalized_texts = normalizer.normalize_batch(texts)

# 方式 4：指定语言
normalized = normalize_text("Hello", language='en')  # 强制英文处理
```

#### `subtitle_utils.py`

关键类和函数：

```python
from utils.subtitle_utils import Text2SRT, generate_srt, generate_srt_clip

# Text2SRT 类构造函数
text2srt = Text2SRT(
    text="hello world",
    timestamp=[[100, 500], [500, 1000]],
    offset=0,
    normalize=True  # 启用正规化 (默认: True)
)

# 获取正规化后的文本
normalized_text = text2srt.text()

# 生成 SRT 格式
srt_output = text2srt.srt(acc_ost=0.0)

# 生成 SRT 字幕
sentence_list = [{
    'text': 'hello world',
    'timestamp': [[100, 500], [500, 1000]],
}]

# 启用正规化
srt_with_norm = generate_srt(sentence_list, normalize=True)

# 禁用正規化
srt_without_norm = generate_srt(sentence_list, normalize=False)
```

## 使用示例

### 示例 1: 基础文本正规化

```python
from utils.text_normalization import normalize_text

# 中文文本
text1 = "我来自深圳，这里的温度约25度。"
print(normalize_text(text1))
# 输出: 我来自深圳，这里的温度约25度。

# 英文文本
text2 = "The price is $99.99 USD!"
print(normalize_text(text2))

# 混合文本
text3 = "2024年3月18日，I am from China."
print(normalize_text(text3))
```

### 示例 2: 在 ASR 流程中使用

```python
from utils.text_normalization import normalize_text

# ASR 识别结果
asr_result = "我来自北京市，今年25岁."

# 正规化文本
normalized = normalize_text(asr_result)

# 保存到文件
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(normalized)
```

### 示例 3: 在 SRT 生成中使用

```python
from utils.subtitle_utils import generate_srt

# ASR 识别结果（句子级）
sentence_list = [
    {
        'text': '我来自深圳',
        'timestamp': [[100, 500], [500, 1000], [1000, 1500]],
    },
    {
        'text': '很高兴见到你',
        'timestamp': [[1500, 2000], [2000, 2500]],
    },
]

# 生成 SRT（自动正规化文本）
srt_content = generate_srt(sentence_list, normalize=True)

# 保存 SRT 文件
with open('subtitles.srt', 'w', encoding='utf-8') as f:
    f.write(srt_content)
```

### 示例 4: 在 videoclipper_v2.py 中使用

```python
# 在现有代码中，当调用 generate_srt 时，确保传入 normalize=True：

# 第一个位置（大约第354行）
res_srt = generate_srt(sentence_info, normalize=True)

# 第二个位置（大约第384行）
res_srt = generate_srt(rec_result[0]['sentence_info'], normalize=True)

# 第三个位置（大约第445行）
srt_clip, subs, srt_index = generate_srt_clip(
    sentence_list, 
    start, 
    end, 
    begin_index=begin_index,
    time_acc_ost=time_acc_ost,
    normalize=True  # 添加此参数
)
```

### 示例 5: 禁用正规化

如果需要保留原始文本不进行正规化：

```python
from utils.subtitle_utils import generate_srt

# 禁用正規化
srt_content = generate_srt(sentence_list, normalize=False)
```

## 文本正规化效果

### 中文文本示例

| 原文本 | 正規化后 | 说明 |
|------|--------|------|
| 我来自北京3号楼 | 我来自北京三号楼 | 数字转中文 |
| 价格$99.99 | 价格九十九点九九美元 | 货币正規化 |
| 今年2024年 | 今年二零二四年 | 年份正規化 |
| 下午3:30开会 | 下午三点三十分开会 | 时间正规化 |

### 英文文本示例

| 原文本 | 正規化后 | 说明 |
|------|--------|------|
| Dr. Smith | Doctor Smith | 缩写展开 |
| $10.99 | ten dollars and ninety nine cents | 货币正規化 |
| 2024 | twenty twenty four | 年份读法 |

## 常见问题

### Q1: WeTextProcessing 安装失败？

**A:** 确保 Python >= 3.7，并尝试：
```bash
pip install --upgrade pip
pip install WeTextProcessing -i https://pypi.org/simple/
```

### Q2: 正规化后文本有误？

**A:** 这可能是 WeTextProcessing 库本身的限制。可以：
1. 检查输入文本的编码
2. 在特定位置禁用正规化：`normalize=False`
3. 提交 issue 到 WeTextProcessing 项目

### Q3: 性能如何？

**A:** 
- 首次加载模型需要几秒钟
- 之后使用单例模式，文本处理速度很快（通常 < 10ms/句）
- 大段文本的正规化是串行进行的

### Q4: 可以关闭正規化吗？

**A:** 可以，在调用时设置 `normalize=False`：
```python
generate_srt(sentence_list, normalize=False)
```

### Q5: 如何在特定位置禁用正規化？

**A:** 修改 Text2SRT 对象的创建时传入 `normalize=False`：
```python
t2s = Text2SRT(text, timestamp, normalize=False)
```

## 技术细节

### 文本处理流程

```
ASR 识别结果
    ↓
process_asr_to_sentence_info()
    ↓
sentence_list (包含 text, timestamp)
    ↓
generate_srt(sentence_list, normalize=True)
    ↓
Text2SRT 对象创建（normalize=True）
    ↓
_apply_normalization() 调用
    ↓
normalize_text() 全局函数
    ↓
TextNormalizer 单例实例
    ↓
ChineseTextNormalizer/EnglishTextNormalizer
    ↓
正规化后的文本
    ↓
SRT 格式输出
    ↓
保存到文件
```

### 语言检测算法

```python
def _detect_language(text: str) -> str:
    """检测文本主要语言"""
    zh_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total_chars = len(text)
    # 中文字符比例 > 30% 判定为中文，否则为英文
    return 'zh' if zh_count / total_chars > 0.3 else 'en'
```

## 运行演示

运行演示脚本查看各种用法：

```bash
cd seperate/video_clip
python demo_text_normalization.py
```

输出示例：
```
╔════════════════════════════════════════════════════════════════════╗
║              WeTextProcessing 文本正规化演示
╚════════════════════════════════════════════════════════════════════╝

======================================================================
演示 1: 基础文本正规化
======================================================================

原文本:  我来自中国，北京市。
正规化:  ...

======================================================================
演示 2: Text2SRT 中的文本正规化
======================================================================
...
```

## 调试和日志

启用调试日志：

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# 现在会看到详细的初始化和处理日志
from utils.text_normalization import normalize_text
normalized = normalize_text("test")
```

## 性能优化建议

1. **使用批量处理**：如果有多个文本，使用 `normalize_batch()` 更高效
2. **重用实例**：使用单例模式 `get_normalizer()`，避免重复初始化
3. **按需禁用**：非关键文本字段可设置 `normalize=False` 跳过处理
4. **缓存结果**：如果同一文本多次出现，可缓存正规化结果

## 后续改进

- [ ] 支持自定义正规化规则
- [ ] 添加正規化结果缓存
- [ ] 支持更多语言
- [ ] 性能基准测试
- [ ] WebService 包装用于远程纠正

## 参考资源

- WeTextProcessing: https://github.com/wenet-e2e/WeTextProcessing
- 中文文本正規化: https://github.com/wenet-e2e/WeTextProcessing/tree/main/tn/chinese
- 英文文本正規化: https://github.com/wenet-e2e/WeTextProcessing/tree/main/tn/english

## 许可和致谢

- WeTextProcessing 库: Apache License 2.0
- 集成代码: 基于项目许可

---

**最后更新**: 2024-03-18
**作者**: 文本处理团队
**版本**: 1.0.0
