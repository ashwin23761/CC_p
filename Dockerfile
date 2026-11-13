FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "cpsim.cli"]
CMD ["run"]
