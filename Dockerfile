FROM python:3.11-slim

WORKDIR /app
COPY . .

# Install the project in editable mode to register the 'server' script
RUN pip install --no-cache-dir -e .

EXPOSE 7860

# Start using the script name defined in pyproject.toml
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
