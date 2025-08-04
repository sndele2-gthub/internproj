# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the main application file with its specific name
COPY app.py .

# Use the correct file name with spaces in the command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
