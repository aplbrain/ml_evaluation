# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

ADD random_performer.py .

# Run random_performer.py when the container launches
CMD [ "python", "random_performer.py" ]
