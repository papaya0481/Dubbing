# Colorful Logger System for IndexTTS2

这个logger系统为IndexTTS2项目提供了美观的彩色控制台输出和进度条显示功能。

## 特性

- 🎨 **彩色输出**: 不同类型的消息使用不同的颜色
- 📊 **层级显示**: 字典和配置信息以树形结构显示
- ⏱️ **时间统计**: 美观的表格显示性能统计信息
- 🎭 **情感向量**: 可视化显示情感向量及其数值
- 📈 **进度条**: 彩色进度条显示推理进度
- 🎯 **多种日志级别**: info, success, warning, error, debug

## 依赖

```bash
pip install rich>=13.7.0
```

## 基本使用

### 1. 导入Logger

```python
from indextts.utils.logger import get_logger, create_progress
```

### 2. 创建Logger实例

```python
logger = get_logger("MyModule")
```

### 3. 基本日志输出

```python
logger.info("这是一条信息")      # [ℹ info] 这是一条信息
logger.success("操作成功!")      # [✓ success] 操作成功!
logger.warning("这是一个警告")    # [⚠ warning] 这是一个警告
logger.error("发生错误!")        # [✗ error] 发生错误!
logger.debug("调试信息")         # [• debug] 调试信息
```

注意：所有日志级别都使用统一的符号+文字标签格式，便于快速识别消息类型。

### 4. 阶段标题

```python
logger.stage("模型加载阶段")
```

### 5. 模型加载信息

```python
logger.model_loaded("GPT Model", "/path/to/checkpoint.pth")
```

### 6. 设备配置信息

```python
logger.device_info(device="cuda:0", is_fp16=True, use_cuda_kernel=False)
```

### 7. 字典显示（层级结构）

```python
config = {
    "model": "IndexTTS2",
    "generation": {
        "temperature": 0.8,
        "top_p": 0.9,
    }
}
logger.print_dict("配置信息", config)
```

### 8. 情感向量显示

```python
# 单个情感向量
emotion_dict = {
    "happy": 0.8,
    "angry": 0.1,
    "sad": 0.2,
    # ... 其他情感
}
logger.print_emotion_vector(emotion_dict)

# 多个情感向量（并列显示）
emotion_vectors = [
    {"happy": 0.9, "angry": 0.1, "sad": 0.0, ...},
    {"happy": 0.2, "angry": 0.7, "sad": 0.3, ...}
]
logger.print_emotion_vector(emotion_vectors)
```

**输出效果：**
- 单个向量：显示为单棵树
- 多个向量：并列显示为多列，每列一个编号的情感向量

这对于比较不同文本段落的情感非常有用！

### 9. 时间统计表格

```python
time_stats = {
    "GPT Generation": 2.5,
    "GPT Forward": 1.2,
    "S2Mel": 3.8,
    "BigVGAN": 1.5,
}
total_time = 9.0
audio_length = 5.0

logger.print_time_stats(time_stats, total_time, audio_length)
```

### 10. 进度条

```python
with create_progress() as progress:
    task_id = progress.add_task("处理中", total=100.0)
    
    for i in range(100):
        # 做一些工作
        progress.update(advance=1, description=f"处理 {i+1}/100")
```

### 11. 面板显示

```python
logger.panel(
    "这是一条重要消息",
    title="注意",
    style="yellow"
)
```

### 12. 分割线

```python
logger.rule("分割线标题")
```

## IndexTTS2集成

在IndexTTS2的`infer_v2.py`中，logger已经被集成：

```python
# 初始化时自动创建
self.logger = get_logger("IndexTTS2")

# 推理过程中的使用示例
self.logger.stage("Starting Inference")
self.logger.info("Processing reference audio...")
self.logger.success("Speech codes generated in 2.5s")
```

## 颜色方案

- **info** (青色): 一般信息
- **success** (绿色): 成功消息
- **warning** (黄色): 警告信息
- **error** (红色): 错误信息
- **debug** (洋红色): 调试信息
- **stage** (蓝色): 阶段标题
- **model** (绿色): 模型加载信息
- **time** (黄色): 时间相关信息
- **value** (青色): 数值显示
- **key** (白色): 键名显示

## 性能统计表格说明

时间统计表格会自动计算：
- 每个阶段的耗时
- 每个阶段占总时间的百分比
- 总推理时间
- 生成音频长度
- **RTF (Real Time Factor)**: 
  - < 1.0 (绿色): 实时性能优秀
  - 1.0-2.0 (黄色): 实时性能良好
  - > 2.0 (红色): 需要优化

## 情感向量可视化

情感向量会以树形结构显示，包含：
- 情感名称
- 可视化条形图（使用█和░字符）
- 精确数值
- 颜色编码：
  - > 0.7: 红色（高强度）
  - 0.4-0.7: 黄色（中强度）
  - 0.1-0.4: 青色（低强度）
  - < 0.1: 暗淡（非常低）

## 运行示例

查看完整示例：

```bash
cd index-tts2
python -m indextts.utils.logger_example
```

## 注意事项

1. **全局Logger**: `get_logger()`会返回全局单例，确保整个应用使用统一的logger实例
2. **时间戳**: 默认启用时间戳，可以在创建时禁用：`ColorfulLogger(enable_timestamp=False)`
3. **进度条**: 使用`with`语句确保进度条正确清理
4. **终端支持**: 需要支持ANSI颜色的终端（大多数现代终端都支持）

## 自定义样式

如果需要自定义颜色和样式，可以修改`logger.py`中的`styles`字典：

```python
self.styles = {
    'info': Style(color="cyan", bold=False),
    'success': Style(color="green", bold=True),
    # ... 添加或修改样式
}
```

## 故障排除

### 颜色不显示
- 确保终端支持ANSI颜色
- 检查是否通过管道重定向输出
- 某些IDE的控制台可能不支持颜色

### 进度条显示异常
- 确保使用`with`语句
- 避免在进度条运行时输出其他内容
- 使用`logger`而不是`print()`

### Rich库导入失败
```bash
pip install rich>=13.7.0
```
