FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (adjust based on your requirements.txt)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libpq-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5001

# Start the app
CMD ["python", "app.py"]
