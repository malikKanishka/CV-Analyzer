# Use Python 3.10 base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /ATS

# Copy core files
COPY . /ATS

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r req.txt
RUN python -m spacy download en_core_web_md

# Expose the Flask default port
EXPOSE 5000

# Command to run the app
CMD ["python", "run.py"]
