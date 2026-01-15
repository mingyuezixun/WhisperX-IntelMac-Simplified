# WhisperX-IntelMac-Simplified 🎙️

**专为 2019 款 Intel Mac 及低显存设备优化的 WhisperX 自动化转录方案。**

## 🌟 核心痛点解决
本项目针对 Intel Mac 运行原版 WhisperX 时的常见问题进行了深度调优，并引入了策略模式以适应不同硬件条件：

- **双模式策略 (New)**：
    - **Full Mode (全量模式)**：使用 WhisperX 官方完整流程（Transcribe -> Align -> Diarize），包含音画对齐，结果最精确，适合性能较好的机器。
    - **Segment Mode (分段模式)**：专为 Intel Mac 优化。将音频物理切片后分段进行说话人识别，彻底解决长时间音频导致的内存溢出 (OOM) 和 CPU 假死问题。
- **智能缓存 (Enhanced)**：转录结果（Transcription）自动缓存为 JSON。如果后续的说话人识别（Diarization）步骤失败，重启程序可直接读取缓存，无需重新转录。缓存文件包含模型名称，防止混淆。
- **性能优化**：自动将音频预处理为 16kHz 单声道 WAV。
- **鲁棒性增强**：忽略 AAC 坏帧，防止 ffmpeg 处理时崩溃。
- **本地化**：集成 `OpenCC` 自动完成繁体转简体。
- **自定义说话人范围**：支持分别设置 `min_speakers` 和 `max_speakers`，解决人数识别不准的问题。
- **HuggingFace Token 支持**：支持传入 HF Token 以使用更先进的 Pyannote 模型 (如 segmentation-3.0)。
- **Docker 化**：一键式环境搭建，无需复杂的本地依赖配置。

---

## 🛠️ 快速开始 (Docker 推荐)

这是最简单的方法，无需在宿主机安装任何 Python 或 PyTorch 依赖。

### 1. 克隆项目
```bash
git clone https://github.com/mingyuezixun/WhisperX-IntelMac-Simplified.git
cd WhisperX-IntelMac-Simplified
```

### 2. 构建环境
```bash
# 这一步会根据 Dockerfile 预装所有依赖，首次运行需等待
docker-compose build
```

### 3. 准备音频
将你的音频文件（例如 `meeting.m4a`）放入项目根目录。

### 4. 启动并运行
**场景 A：使用分段模式 (推荐 Intel Mac 使用)**
适合长音频（>1小时），通过切片降低内存压力。
```bash
# 启动容器
docker-compose up -d

# 运行转录 (使用 segment 模式，每 10 分钟切一片)
docker exec -it whisper_offline python3 main.py --file "meeting.m4a" --mode segment --chunk 10 --num 2
```

**场景 B：使用全量模式**
适合短音频或性能较好的机器，包含精确的单词级对齐。
```bash
docker exec -it whisper_offline python3 main.py --file "meeting.m4a" --mode full --model medium --num 2
```

结果将生成在 `output/` 文件夹下。

---

## ⚙️ 命令行参数说明

| 参数 | 缩写 | 说明 | 默认值 | 示例 |
| :--- | :--- | :--- | :--- | :--- |
| `input` | (位置参数) | **(必填)** 原始音频文件路径 | 无 | `meeting.mp3` |
| `--mode` | 无 | **识别模式**：`full` 或 `segment` | `full` | `--mode segment` |
| `--model` | `-m` | Whisper 模型大小 (tiny, small, medium, large-v2 等) | `small` | `-m large-v2` |
| `--num` | `-n` | 预期的说话人数量 (辅助聚类) | `2` | `-n 4` |
| `--chunk` | `-c` | **分段时长(分钟)**，仅在 `segment` 模式下有效 | `10` | `-c 15` |
| `--language` | `-l` | 识别语言代码 | `zh` | `-l en` |
| `--min_speakers` | 无 | **(可选)** 最少说话人数 (覆盖 -n) | 无 | `--min_speakers 2` |
| `--max_speakers` | 无 | **(可选)** 最多说话人数 (覆盖 -n) | 无 | `--max_speakers 5` |
| `--hf_token` | 无 | HuggingFace Token (用于加载更强模型) | 无 | `--hf_token hf_...` |

---

## 🧠 模式详解

### 1. Full Mode (全量模式)
- **流程**：`Transcribe` -> `Align` (强制) -> `Diarize` -> `Assign Speakers`
- **优点**：包含 `Align` 步骤，即使语速很快也能精确匹配每个单词的说话人；生成的字幕时间轴非常精准。
- **缺点**：`Align` 和 `Diarize` 步骤对内存需求较高，在长音频上容易导致 Intel Mac 崩溃。

### 2. Segment Mode (分段模式)
- **流程**：`Transcribe` -> `Segmented Diarize` (按 Chunk 切分处理) -> `Merge`
- **优点**：极低内存占用。通过物理切分音频，让 Diarization 每次只处理一小段（如 10 分钟），几乎可以在任何机器上跑完长达数小时的录音。
- **缺点**：跳过了 `Align` 步骤，说话人匹配基于时间戳区间映射，精度略低于 Full Mode（但在长会议场景通常足够）。

---

## 📂 目录结构与输出
```Plaintext
WhisperX-IntelMac-Simplified/
├── main.py                # 入口文件
├── ...
└── output/                # 结果输出目录
    ├── meeting.wav        # 预处理后的音频
    ├── meeting_transcribe_small.json  # [缓存] 转录中间结果 (含模型名)
    ├── meeting_result.txt # 最终带说话人的文本
    ├── meeting.srt        # 字幕文件
    └── ...
```

## 💡 常见问题 (FAQ)

**Q: 程序运行了一半报错了，需要重头开始吗？**
A: 不需要。程序会自动保存转录阶段的 JSON 缓存（如 `xxx_transcribe_small.json`）。再次运行相同命令时，会检测并加载缓存，直接跳过最耗时的转录步骤，继续进行说话人识别。

**Q: Intel Mac 依然过热或卡顿？**
A: 建议使用 `--mode segment` 并减小 `--chunk` 值（例如设为 5）。同时确保 Docker 分配了足够的 CPU 资源。

## 📜 协议
MIT License
