FROM python:3.11-slim

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 user
USER user

# Set working directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Copy and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Expose HF Spaces port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]