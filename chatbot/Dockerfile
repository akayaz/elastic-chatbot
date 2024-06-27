FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PORT=8501

CMD streamlit run advanced-chatbot-rerank.py --server.port=${PORT} --browser.serverAddress="0.0.0.O"

#CMD ["streamlit", "run", "advanced-chatbot-rerank.py", "--server.enableCORS", "false", "--browser.serverAddress", "0.0.0.0", "--browser.gatherUsageStats", "false"]