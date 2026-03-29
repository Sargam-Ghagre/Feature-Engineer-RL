FROM python:3.11-slim

WORKDIR /app

# 1. Copy files as root
COPY . .

# 2. Install dependencies as root
RUN pip install --no-cache-dir \
    pandas \
    "numpy<2.0.0" \
    scikit-learn \
    fastapi \
    uvicorn \
    gymnasium \
    stable-baselines3 \
    shimmy \
    pydantic \
    openenv-core \
    httpx

# 3. Create user and fix permissions WHILE STILL ROOT
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# 4. Now switch to the restricted user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
