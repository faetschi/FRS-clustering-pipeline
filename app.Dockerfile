# Dockerfile.app
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy app code
COPY app /app

# Expose any app-specific ports if needed
EXPOSE 8000

# Run your app (adjust entrypoint as needed)
CMD ["python", "fr_clustering.py"]
