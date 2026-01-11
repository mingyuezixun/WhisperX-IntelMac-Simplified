FROM python:3.10-slim

# 1. 将 Debian 软件源更换为阿里云（针对 Debian 12 / bookworm 优化）
# 注意：python:3.10-slim 基于 Debian 12 (bookworm)，阿里云源地址需匹配
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main non-free non-free-firmware contrib" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main non-free non-free-firmware contrib" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-backports main non-free non-free-firmware contrib" >> /etc/apt/sources.list

# 2. 安装 ffmpeg 和 git
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# 3. 设置 Python 国内镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装核心依赖
# 【关键修改】锁死 Torch 版本为 2.6.0，防止未来更新导致代码不兼容
RUN pip install --no-cache-dir \
    "numpy<2" \
    "torch==2.6.0" \
    "torchaudio==2.6.0"

# 5. 安装 WhisperX
RUN pip install git+https://github.com/m-bain/whisperX.git

# 6. 安装其他工具库
# 【关键修改】在这里补上了 pydub
RUN pip install --no-cache-dir \
    pyannote.audio \
    huggingface_hub \
    pandas \
    opencc-python-reimplemented \
    pydub

# 7. 设置工作目录
WORKDIR /app
