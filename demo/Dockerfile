# CrisisLens HF Space (Docker, free CPU tier): llama.cpp server (official release) + FastAPI web console.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ARG LLAMA_TAG=b12052
RUN curl -sSL -o /tmp/llama.tgz https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz \
 && mkdir -p /opt/llama && tar -xzf /tmp/llama.tgz -C /opt/llama && rm /tmp/llama.tgz \
 && find /opt/llama -name llama-server -exec chmod +x {} \;
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 user && chown -R user /app
USER user
ENV CRISISLENS_BASE_URL=http://127.0.0.1:8080/v1 CRISISLENS_MODEL=crisislens
EXPOSE 7860
CMD ["bash", "start.sh"]
