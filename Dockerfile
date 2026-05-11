# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HF_HOME /tmp/huggingface

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Spacy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Ensure the models and scaler are in the correct place (root)
# They are already copied by 'COPY . .'

# Hugging Face Spaces expects the app to run on port 7860
ENV PORT 7860
EXPOSE 7860

# Production settings
ENV DEBUG False

# Run database migrations and collect static files
RUN python manage.py migrate
RUN python manage.py collectstatic --noinput

# Use gunicorn to serve the Django app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "sayardesk.wsgi:application"]
