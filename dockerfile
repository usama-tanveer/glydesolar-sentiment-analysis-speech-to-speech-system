# Use a minimal Python image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install system dependencies (including ffmpeg, but no CUDA or GPU dependencies)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean

# Install Python dependencies without GPU libraries
# Use CPU-only variants for frameworks like TensorFlow or PyTorch
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install python-multipart

# Copy the rest of the project files into the container
COPY . /app/

# Expose the port your application runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
