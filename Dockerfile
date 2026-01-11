FROM python:3.10-slim

# 1. 【新增】将 Debian 软件源更换为阿里云（针对 Debian 12 / trixie 优化）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 2. 现在再安装 ffmpeg 就会非常快
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# 3. 设置 Python 国内镜像源 (这一步你已经有了)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装大型依赖（分开写，方便利用缓存）
RUN pip install --no-cache-dir "numpy<2" "torch>=2.0.0" "torchaudio"
RUN pip install git+https://github.com/m-bain/whisperX.git
RUN pip install pyannote.audio huggingface_hub pandas opencc-python-reimplemented

# 5. 设置工作目录
WORKDIR /app
