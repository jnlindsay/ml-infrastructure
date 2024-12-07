# Use a Python base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Command to run on container startup
CMD ["tail", "-f", "/dev/null"]