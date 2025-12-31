# Use the official Unsloth image which comes with PyTorch and CUDA pre-installed
# Check https://hub.docker.com/r/unsloth/unsloth for latest tags
FROM unsloth/unsloth:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Note: unsloth is already installed in the base image, but we install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose Gradio port
EXPOSE 7860

# Set the default command to run the Gradio app
# Users can override this with "python scripts/cli.py ..." for headless mode
CMD ["python", "scripts/app.py"]
