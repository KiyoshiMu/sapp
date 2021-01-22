FROM python:3.8-slim-buster
COPY dist/*whl app.py index.bin *onnx cases.json /app/
WORKDIR /app
RUN pip install --upgrade pip setuptools wheel && pip install *whl
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
