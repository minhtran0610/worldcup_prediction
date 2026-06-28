FROM python:3.13-slim

# Use host network during build so pip can reach PyPI/PyTorch without DNS issues
ARG BUILDKIT_NETWORK=host

WORKDIR /app

# CPU-only PyTorch — avoids pulling the ~5 GB CUDA runtime
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Scheduler-specific extras not in pyproject.toml
RUN pip install --no-cache-dir "apscheduler>=3.10,<4" requests

# Project dependencies (torch already installed above, --no-deps skips reinstall)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    pandas scipy scikit-learn lightgbm xgboost typer rich matplotlib \
    soccerdata pyarrow

# Copy source and install package without re-resolving deps
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "docker/scheduler.py"]
