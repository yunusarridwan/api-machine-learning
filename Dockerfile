FROM python:3.11

WORKDIR /api_ml_tflite

# Copy requirements.txt separately to leverage Docker cache
COPY requirements.txt /api_ml_tflite/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install AVX2 and FMA dependencies
RUN apt-get update && apt-get install -y gcc g++ libatlas-base-dev

# Install the necessary OpenGL library
RUN apt-get install -y libgl1-mesa-glx

# Copy the rest of the application code
COPY . /api_ml_tflite

# Install dependencies including tf_slim
RUN pip install tf_slim

EXPOSE 8080

CMD ["python", "server.py"]
