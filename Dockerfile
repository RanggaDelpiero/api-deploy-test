FROM python:3.9.6

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Download the key.json file using wget
#RUN apt-get update && apt-get install -y wget
#RUN wget $JSON_KEY

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir  -r requirements.txt

# Copy the key.json file to the working directory
# COPY key.json .

# Copy the application code
COPY . .

ARG KEY_JSON
ENV KEY_JSON=$KEY_JSON

ENV PORT 5000
CMD exec gunicorn --bind :$PORT main:app --workers 1 --threads 1 --timeout 60
