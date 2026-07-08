FROM python:3.10-slim

WORKDIR /app

COPY frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY frontend /app/frontend

EXPOSE 8501

ENV API_URL=http://backend:8000

CMD ["streamlit", "run", "/app/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]