FROM python:3.8-slim

WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    default-jdk \
    build-essential \
    python3-dev \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    git \
    curl \
    g++ \
    gcc \
    && apt clean && rm -rf /var/lib/apt/lists/*

#JAVA_HOME 환경변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

#작업 디렉토리 설정
WORKDIR /app

# 파이썬 패키지 설치
RUN apt-get update && apt-get install -y gcc g++ gfortran
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#Tokenizer sentencepiece 설치
RUN pip install --no-cache-dir sentencepiece huggingface_hub[hf_xet]

# 로컬 PyKoSpacing 폴더 복사 및 설치 (수정된 setup.py가 반드시 포함)
COPY PyKoSpacing /app/PyKoSpacing
RUN pip install --no-cache-dir /app/PyKoSpacing

# 앱 소스 코드 복사
COPY app ./app

# 메일 데이터 및 인덱스 파일 복사
COPY src/email_embeddings.npy /app/src/
COPY src/email_texts.npy /app/src/
COPY src/email_index.faiss /app/src/
COPY src/Mails_trimmed1.csv /app/src

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]