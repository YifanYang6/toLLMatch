FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
RUN apt update && apt install python3.10-venv ffmpeg -y

COPY . /app/toLLMatch
#RUN git clone https://github.com/YifanYang6/toLLMatch.git /app/toLLMatch
WORKDIR /app/toLLMatch

RUN touch ./.env
ENV VLLM_SERVER_ENDPOINT_URL=http://localhost:8001/generate
ENV HF_HOME=/app/toLLMatch/hf_cache
ENV ASR_SERVER_ENDPOINT_URL=http://localhost:8002
#ENV LLM_MODEL=meta-llama/Meta-Llama-3.1-8B
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$PIP_INDEX" && \
    python -m pip install --upgrade pip
# SETUP
WORKDIR /app/toLLMatch/evaluation
RUN pip install -e ../SimulEval \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    python-dotenv protobuf==3.20.2 termcolor==2.4.0 accelerate \
    fastapi-cli==0.0.2 editdistance==0.8.1 jax[cuda12]==0.4.28 \
    msgpack==1.0.8 chardet==5.2.0 nltk==3.8.1\
    openai==1.23.1 vllm==0.4.1 \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.3/flash_attn-2.5.3+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
### vllm
#RUN python3.10 -m venv .venv_llm
#RUN source .venv_llm/bin/activate &&\
#    pip config set global.index-url "$PIP_INDEX" && \
#    pip config set global.extra-index-url "$PIP_INDEX" && \
#    python -m pip install --upgrade pip && \
#    pip install vllm
## ssh
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建 SSH 运行目录
RUN mkdir /var/run/sshd


# 允许 root 用户通过 SSH 登录
RUN sed -i 's/^#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 允许密码认证登录
RUN sed -i 's/^#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 暴露 SSH 端口
EXPOSE 22
WORKDIR /app

# 创建启动脚本
RUN cat <<EOF > /start.sh
#!/bin/bash
echo "Starting SSH, Jupyter Notebook"
service ssh start
jupyter notebook --port=8888 --allow-root
EOF

# 赋予执行权限
RUN chmod +x /start.sh

# 使用脚本启动服务
CMD ["/bin/bash", "/start.sh"]

