# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY  ./ ./

# Expose the port on which the application runs
EXPOSE 8000

CMD ["flask", "run", "--host", "0.0.0.0" ]