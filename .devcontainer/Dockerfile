# Use NVIDIA's PyTorch container with CUDA 12.6 and PyTorch 2.5.0-alpha
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set the working directory in the container
WORKDIR /workspace

# Copy your project files to the container
COPY . /workspace

# Install any additional dependencies required by your project
RUN pip install --no-cache-dir -r .devcontainer/requirements.txt

# Expose port 8000 (or whatever port your application uses)
EXPOSE 8000

# Set the default command to run your Python application
CMD ["python3", "main.py"]  # Adjust the script name to your project
