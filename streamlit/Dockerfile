FROM python:3.10-slim

WORKDIR /app

COPY streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit ./streamlit
COPY models ./models

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0"]
