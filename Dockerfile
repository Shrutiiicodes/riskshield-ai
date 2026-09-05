FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the pipeline once at build time isn't ideal for real data changes,
# so we expect evaluation/reports/ to be generated via `docker compose run pipeline`
# before starting the API/dashboard services.

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
