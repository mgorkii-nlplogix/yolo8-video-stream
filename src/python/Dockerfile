# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the inference folder to /app/inference
COPY inference /app/inference

# Update pip and install requirements
RUN pip install --upgrade pip
RUN pip install  -r inference/requirements.txt

# Azure FileShare will be mounted to that location
# This will be done by container app job. Configured in Bicep
# Not sure yet if we need that
RUN mkdir /mnt/models
# Change mode to read/write/execute for everyone
RUN chmod -R 777 /mnt/models
# Move to the code repo
WORKDIR /app/inference

# Set the entrypoint to run inference.py
ENTRYPOINT ["python3.9", "inference.py"]