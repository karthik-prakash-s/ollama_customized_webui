# Use the official Python image as a base
FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Flask and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY src/ .

# Expose the port on which the Flask app will run
EXPOSE 8000

# Command to run the Flask application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
