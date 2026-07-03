FROM python:3.11-slim

WORKDIR /app

# System deps for common scientific/python wheels (kept minimal)
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY pyproject.toml /app/pyproject.toml
COPY alembic.ini /app/alembic.ini
COPY migrations /app/migrations
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

