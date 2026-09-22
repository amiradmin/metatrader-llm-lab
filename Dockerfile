FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "pydantic>=2.9.0" \
    "uvicorn>=0.30.0"

EXPOSE 8010

CMD ["uvicorn", "metatrader_llm_lab.bridge.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8010"]
