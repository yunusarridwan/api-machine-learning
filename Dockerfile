FROM python:3.11

WORKDIR /api_ml_tflite

COPY . /api_ml_tflite

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "server.py"]
