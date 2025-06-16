# Stage 1: Build environment
FROM python:3.10.6-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Minimal runtime image
FROM python:3.10.6-slim

WORKDIR /prod

# Install runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app code
COPY sentiscope sentiscope
COPY models models

# Copy nltk data
COPY nltk_data /usr/share/nltk_data
ENV NLTK_DATA=/usr/share/nltk_data

# Avoid running as root
RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "sentiscope.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
