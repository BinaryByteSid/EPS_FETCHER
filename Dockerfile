FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and database
COPY backend ./backend
COPY frontend ./frontend
COPY ["ISIN'S.xlsx", "./ISIN'S.xlsx"]

EXPOSE 8000

# Start command — bind to Render's $PORT if provided, else default to 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
