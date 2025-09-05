FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask-SocketIO port
EXPOSE 8080

# Run the app with Python (not flask run)
CMD ["python", "app.py"]
