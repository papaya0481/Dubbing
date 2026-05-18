# V2C 电影数据集构建工具

本目录包含两个用于从电影视频中提取音视频片段、构建数据集的脚本。

---

## 脚本概览

| 脚本 | 用途 |
|------|------|
| `build_movie_origin.py` | 按**单条字幕**逐条裁剪视频/音频，生成原始片段数据集 |
| `build_movie_dataset.py` | 用**滑动窗口**将连续多条字幕组合为样本（需含情感变化），生成面向情感识别任务的数据集 |

---

## 输入数据格式

### 1. `--movie-map`：电影信息映射表（CSV）

每行对应一部电影，必须包含以下列：

| 列名 | 类型 | 是否必需 | 说明 |
|------|------|----------|------|
| `movie` | string | **必须** | 电影唯一标识名（与 utterances CSV 中的 `movie` 列对应） |
| `filename` | string | **必须** | 视频文件名（相对于 `--video-root`）或绝对路径 |
| `checked` | bool | **必须** | 是否经过人工校验，必须为 `True` 才会被处理 |
| `time_offset` | float | 建议填写 | 字幕与视频的时间偏移（秒），正值表示视频比字幕滞后，填 `0.0` 表示无偏移；若缺失该列则默认为 `0.0` |

**示例：**
```csv
movie,filename,checked,time_offset
Zootopia,Zootopia.mkv,True,1.5
Coco,Coco.mp4,True,0.0
Moana,Moana.mkv,False,
```
> `checked=False` 或 `time_offset` 为空（NaN）的行会被跳过并输出警告。

---

### 2. `--utterances`：字幕/话语记录表（CSV）

经过转换的需要裁切的片段。

**示例：**
```csv
movie,speaker,utterance,emotion,emotion_id,start_time,end_time,srt_index
Zootopia,Judy,I'm going to make the world a better place.,joy,1,00:01:12,500,00:01:15,200,42
Zootopia,Judy,Just you wait.,neutral,0,00:01:15,800,00:01:17,000,43
Zootopia,Nick,Oh I'm waiting.,anger,3,00:01:17,500,00:01:19,300,44
```

---

## 脚本使用说明

### 步骤

先使用`build_movie_origin.py`进行逐条字幕裁剪，生成原始片段数据集；再使用`build_movie_dataset.py`在此基础上进行滑动窗口组合，生成最终的数据集。

使用`build_movie_origin.py`进行逐条字幕裁剪时，先使用`--movie` 标签选择单部电影进行测试（即下面的，确认时间偏移等参数设置正确后再批量处理所有电影，避免不必要的重复裁剪。

随后拥有准确的时间信息后，可以在`--movie-map`输入的那个csv里，将`checked`列设置为`True`，表示已经校验过了，之后就可以直接批量处理所有电影了。

然后再使用`build_movie_dataset.py`进行滑动窗口组合，生成最终的数据集。超参数可以不需要调整。

**注意：两个脚本的输出目录不要一样，方便后续分开**。


### `build_movie_origin.py`：逐条字幕裁剪

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--movie-map` | string | **是** | `movie_video_map.csv` 路径 |
| `--utterances` | string | **是** | 字幕记录 CSV 路径 |
| `--video-root` | string | **是** | 存放电影视频文件的根目录 |
| `--output-root` | string | **是** | 输出目录（会自动创建） |
| `--test` | flag | 否（调试用） | 测试模式：只处理前 4 部电影 |
| `--no-offset` | flag | 否（调试用） | 忽略 `time_offset`，直接使用字幕原始时间 |
| `--movie` | string | 否 | 只处理指定的某一部电影（按 `movie` 名称筛选） |

#### 使用示例

```bash
# 完整运行
python build_movie_origin.py \
    --movie-map /data/movie_video_map.csv \
    --utterances /data/movie_speaker_emotion.csv \
    --video-root /data/videos \
    --output-root /data/output_origin

# 测试模式（只处理前4部电影）
python build_movie_origin.py \
    --movie-map /data/movie_video_map.csv \
    --utterances /data/movie_speaker_emotion.csv \
    --video-root /data/videos \
    --output-root /data/output_origin \
    --test

# 只处理某一部电影
python build_movie_origin.py \
    --movie-map /data/movie_video_map.csv \
    --utterances /data/movie_speaker_emotion.csv \
    --video-root /data/videos \
    --output-root /data/output_origin \
    --movie Zootopia

```

#### 输出结构

```
output_origin/
├── videos/                          # 视频片段 (.mp4)
│   ├── Zootopia_Judy_1_42.mp4
│   └── ...
├── audios/
│   └── ost/                         # 音频片段 (.wav, 16kHz 单声道) 及文本 (.txt)
│       ├── Zootopia_Judy_1_42.wav
│       ├── Zootopia_Judy_1_42.txt
│       └── ...
├── meta_files/                      # 每部电影的元数据 CSV
│   ├── Zootopia.csv
│   └── ...
└── metadata.csv                     # 所有电影合并的完整元数据
```

**文件命名规则：** `{movie}_{speaker}_{sample_id}_{srt_index}.mp4/wav/txt`

---

### `build_movie_dataset.py`：滑动窗口组合样本

在逐条裁剪的基础上，将同一说话人的连续若干条字幕组合为一个样本，要求样本内**包含至少一次情感变化**。

#### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--movie-map` | string | **是** | — | `movie_video_map.csv` 路径 |
| `--utterances` | string | **是** | — | 字幕记录 CSV 路径 |
| `--video-root` | string | **是** | — | 存放电影视频文件的根目录 |
| `--output-root` | string | **是** | — | 输出目录（会自动创建） |
| `--max-gap` | float | 否（无需调整） | `10.0` | 同一样本内相邻字幕间允许的最大时间间隔（秒），超过则不合并 |
| `--max-words` | int | 否（无需调整） | `50` | 长度为 3 的样本允许的最大总词数，超过则舍弃该 3 条组合 |
| `--test` | flag | 否（不需要开启） | — | 测试模式：只处理前 4 部电影 |
| `--prefer-longer` | flag | 否（不需要开启） | — | 若长度为 3 的样本有效，则跳过同起始位置的长度为 2 的样本，避免冗余 |
| `--no-offset` | flag | 否（不需要开启） | — | 忽略 `time_offset`，直接使用字幕原始时间 |

### 使用参考
直接参考`make.sh`中的命令行示例，先运行`build_movie_origin.py`，确认时间偏移等参数设置正确后再批量处理所有电影；随后运行`build_movie_dataset.py`生成最终数据集。超参数可以不需要调整。

#### 使用示例

```bash
# 完整运行（默认参数）
python build_movie_dataset.py \
    --movie-map /data/movie_video_map.csv \
    --utterances /data/movie_speaker_emotion.csv \
    --video-root /data/videos \
    --output-root /data/output_dataset

# 严格间隔控制 + 优先长样本
python build_movie_dataset.py \
    --movie-map /data/movie_video_map.csv \
    --utterances /data/movie_speaker_emotion.csv \
    --video-root /data/videos \
    --output-root /data/output_dataset \
    --max-gap 5.0 \
    --max-words 40 \
    --prefer-longer

# 测试模式
python build_movie_dataset.py \
    --movie-map /data/movie_video_map.csv \
    --utterances /data/movie_speaker_emotion.csv \
    --video-root /data/videos \
    --output-root /data/output_dataset \
    --test
```

#### 输出结构

```
output_dataset/
├── videos/                                        # 视频片段 (.mp4)
│   ├── Zootopia_sample_1_42-43.mp4
│   └── ...
├── audios/
│   └── ost/                                       # 音频片段 (.wav, 16kHz 单声道) 及文本 (.txt)
│       ├── Zootopia_sample_1_42-43.wav
│       ├── Zootopia_sample_1_42-43.txt            # 合并后的台词文本（空格连接）
│       └── ...
├── meta_files/                                    # 每部电影的元数据 CSV
│   ├── Zootopia.csv
│   └── ...
└── metadata.csv                                   # 所有电影合并的完整元数据
```

**文件命名规则：** `{movie}_sample_{sample_id}_{first_srt_index}-{last_srt_index}.mp4/wav/txt`

---

## 输出 metadata.csv 字段说明

### `build_movie_origin.py` 输出字段

| 字段 | 说明 |
|------|------|
| `Movie` | 电影名称 |
| `Speaker` | 说话人 |
| `Emotion` | 情感标签 |
| `Emotion_ID` | 情感标签 ID |
| `Srt_Index` | 字幕序号 |
| `Utterance` | 原始台词 |
| `Start_Time_Original` | 原始字幕开始时间 |
| `End_Time_Original` | 原始字幕结束时间 |
| `Start_Time_Adjusted` | 应用偏移后的开始时间 |
| `End_Time_Adjusted` | 应用偏移后的结束时间 |
| `Time_Offset` | 实际使用的时间偏移值（秒） |
| `Clip_Filename` | 视频片段文件名（为空表示裁剪失败） |
| `Clip_Path` | 视频片段相对路径 |
| `Audio_Filename` | 音频片段文件名（为空表示提取失败） |
| `Audio_Path` | 音频片段相对路径 |

### `build_movie_dataset.py` 输出字段

| 字段 | 说明 |
|------|------|
| `Movie` | 电影名称 |
| `Speaker` | 说话人 |
| `Emotions` | 各字幕情感标签列表，如 `['neutral', 'joy']` |
| `Utterances` | 各字幕台词列表 |
| `Length` | 样本包含的字幕条数（2 或 3） |
| `Srt_Indices` | 各字幕的 SRT 序号列表 |
| `Start_Time_Original` | 第一条字幕原始开始时间 |
| `End_Time_Original` | 最后一条字幕原始结束时间 |
| `Start_Time_Adjusted` | 应用偏移后的开始时间 |
| `End_Time_Adjusted` | 应用偏移后的结束时间 |
| `Time_Offset` | 实际使用的时间偏移值（秒） |
| `Clip_Filename` | 视频片段文件名（为空表示裁剪失败） |
| `Clip_Path` | 视频片段相对路径 |
| `Audio_Filename` | 音频片段文件名（为空表示提取失败） |
| `Audio_Path` | 音频片段相对路径 |

---

## 依赖环境

- Python 3.7+
- `pandas`
- `tqdm`
- `ffmpeg`（需已安装并可在命令行调用）

```bash
pip install pandas tqdm
# ffmpeg 安装（Ubuntu）
sudo apt install ffmpeg
```

---

## 常见问题

**Q：某部电影被跳过，日志显示 `checked != True`？**  
A：检查 `movie_video_map.csv` 中对应行的 `checked` 列，确保值为字符串 `True`（注意大小写）。

**Q：字幕时间与视频不同步？**  
A：在 `movie_video_map.csv` 中调整 `time_offset` 字段。正值表示视频播放比字幕晚（字幕超前），负值表示字幕滞后。确认后，去掉 `--no-offset` 参数正常运行即可。

**Q：视频/音频裁剪失败（输出字段为空）？**  
A：检查 `ffmpeg` 是否正确安装，以及视频文件路径是否正确。日志会输出具体的 `warning` 信息。
