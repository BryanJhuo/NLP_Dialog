FROM python:3.10-slim

# 安裝必要套件
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 建立虛擬環境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安裝 Python 套件
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 複製原始碼
COPY . .

# 啟動應用程式（這裡你之後可以改成 streamlit、flask、測試程式等）
CMD ["python", "app/main.py"]
