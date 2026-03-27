FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcribe.py .

# Mount your audio/video files here at runtime
VOLUME ["/audio"]

ENTRYPOINT ["python", "transcribe.py"]
