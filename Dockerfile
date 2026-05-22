FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch vllm tokenizer for newer transformers compatibility
RUN sed -i 's/tokenizer.all_special_tokens_extended)/getattr(tokenizer, "all_special_tokens_extended", []))/g' \
    /usr/local/lib/python3.11/dist-packages/vllm/transformers_utils/tokenizer.py

# Copy handler code
COPY handler.py .

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]
