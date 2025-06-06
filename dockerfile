FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./app /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
