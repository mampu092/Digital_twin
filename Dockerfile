FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
CMD ["python","-m","streamlit","run","dashboard/app.py","--server.address=0.0.0.0","--server.port=8501"]
