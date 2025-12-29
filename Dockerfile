# Dockerfile cho Fly.io - Telegram Bot Worker
FROM python:3.11-slim

# Cài đặt dependencies hệ thống
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements trước để cache layer
COPY requirements.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY 2_crypto.py .

# Chạy bot
CMD ["python", "2_crypto.py"]

