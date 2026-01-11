# WhisperX-IntelMac-Simplified 🎙️

**专为 2019 款 Intel Mac 优化的 WhisperX 自动化转录方案。**



## 🌟 核心痛点解决
本项目针对 Intel Mac 运行原版 WhisperX 时的常见问题进行了深度调优：
- **性能优化**：自动将音频预处理为 16kHz 单声道 WAV，显著降低 CPU 负载。
- **鲁棒性增强**：忽略 AAC 坏帧，防止转录中途崩溃。
- **智能缓存**：引入 JSON 缓存机制，若说话人识别（Diarization）失败，重启后可跳过转录直接继续。
- **本地化**：集成 `OpenCC` 自动完成繁简转换。
- **环境隔离**：提供一键式 Docker 方案，免去繁琐的 PyTorch 和 FFmpeg 环境配置。

---

## 🛠️ 快速开始 (Docker 推荐)

这是最简单的方法，无需在宿主机安装任何 Python 依赖。

### 1. 克隆项目
```bash
git clone [https://github.com/mingyuezixun/WhisperX-IntelMac-Simplified.git](https://github.com/mingyuezixun/WhisperX-IntelMac-Simplified.git)
cd WhisperX-IntelMac-Simplified
```
### 2. 构建环境
```bash
# 这一步会根据 Dockerfile 预装所有依赖，仅需执行一次
docker-compose build
```
### 3. 启动容器并执行
将你的音频文件（如 meeting.aac）放入项目根目录，然后：
```bash
# 启动容器
docker-compose up -d

# 运行转录命令（结果将生成在 output/ 文件夹下）
docker exec -it whisper_offline python3 main.py --file "meeting.aac" --num 2
```
📂 目录结构说明
```Plaintext
WhisperX-IntelMac-Simplified/
├── main.py                # 主程序：逻辑处理核心
├── Dockerfile             # 镜像构建文件（含清华源加速）
├── docker-compose.yml     # 容器编排文件
├── requirements.txt       # Python 依赖清单
├── .gitignore             # 自动忽略大文件和缓存
└── output/                # [自动生成] 存放 WAV、JSON 缓存及最终转录结果
```
⚙️ 命令行参数

|参数|说明|默认值|
| ----- | ------- |------- |
|--file|(必填) 原始音频路径|无|
|--num|预期的说话人数量|2|
|--lang|识别语言|zh|

💡 Intel Mac 运行贴士
- 1.散热建议：Intel Mac 在进行 Diarization 阶段 CPU 会满载，建议接通电源并保持散热通畅。

- 2.离线模式：如果你已经将模型下载到 ~/.cache，本项目已配置自动挂载，可实现真正的离线运行。

- 3.输出结果：所有结果将自动生成在 output/ 文件夹下，根目录保持整洁。

📜 协议
MIT License
