# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only the requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install only necessary packages, no cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app
COPY . .

# Expose port for Koyeb
EXPOSE 8080

# Set the command to run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
