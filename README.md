# WhisperX-IntelMac-Simplified 🎙️

**针对 2019 款 Intel Mac 深度优化的 WhisperX 自动化转录方案。**



## 🌟 为什么选择这个方案？
在 Intel Mac（如 2019 MacBook Pro）上运行 WhisperX 往往面临：CPU 过热、内存溢出（OOM）、AAC 坏帧导致崩溃、输出繁体中文等痛点。本项目通过以下技术手段解决了这些问题：

-   **音频预处理**：自动调用 FFmpeg 将采样率降至 16kHz 并转为单声道 WAV，大幅减轻 CPU 压力。
-   **鲁棒性修复**：强制忽略 AAC 损坏帧，防止转录中途报错。
-   **智能缓存系统**：引入 JSON 缓存机制。若在说话人识别阶段崩溃，重启后直接读取缓存，无需重复转录。
-   **中文本地化**：集成 `OpenCC` 自动繁简转换，并支持非 ASCII 中文 JSON 导出。
-   **一键式体验**：从环境安装到命令行传参处理，全流程自动化。

## 🛠️ 环境要求
1.  **FFmpeg**: 必须在系统级安装。
    ```bash
    brew install ffmpeg
    ```
2.  **Python**: 建议版本 3.10+。
3.  **Docker (可选)**: 建议在 Docker 容器内运行以隔离环境。

## 🚀 安装与运行

### 1. 克隆仓库
```bash
git clone [https://github.com/你的用户名/WhisperX-IntelMac-Simplified.git](https://github.com/你的用户名/WhisperX-IntelMac-Simplified.git)
cd WhisperX-IntelMac-Simplified
```

### 2. 安装依赖脚本内置了自动安装逻辑，你也可以手动安装：
```bash
pip install -r requirements.txt
```
### 3. 一键转录将音频文件（如 meeting.aac）放入项目根目录，运行：
```bash
python main.py --file "meeting.aac" --num 2
```
⚙️ 命令行参数说明参数说明
参数,说明,默认值
--file,(必填) 原始音频路径,无
--num,预期的说话人数量,2
--lang,识别语言,zh

📂 项目结构
- main.py: 主程序脚本。
- requirements.txt: Python 依赖清单。
- .gitignore: 已配置自动忽略音频、模型权重及 JSON 缓存。
- *_16k_mono.wav: 自动生成的预处理音频（运行后产生）。
- *_transcribe.json: 自动生成的转录缓存（支持中文查看）。
- *_结果.txt: 最终的简体中文转录稿。

⚠️ Intel Mac 运行建议
- 性能预期：对于 1 小时音频，Intel Mac 的总处理时间约为 1 小时左右（1:1）。
- 散热提示：说话人识别（Diarization）阶段 CPU 负载极高，请确保通风良好，接通电源。
- 内存优化：本项目默认使用 int8 计算类型，有效避免 16GB 内存机型崩溃。

⚖️ 协议
- MIT License
