FROM python:3.6.8

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install -r requirements.txt

EXPOSE 8501

COPY . .

CMD streamlit run Liver-Health-Analyzer-App.py