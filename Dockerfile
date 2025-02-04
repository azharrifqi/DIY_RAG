FROM python:3.9-slim

RUN apt-get update && apt-get install
# RUN apt-get update
# RUN apt-get install git
WORKDIR /app
# COPY ./prod/ /app
COPY ./ /app

RUN pip3 install -r requirements.txt

EXPOSE 8000
# # CMD [ "executable" ]
# ENTRYPOINT ["streamlit", "run", "demo_1_backup_feedback.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]
# ENTRYPOINT [ "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000" ]

# EXPOSE 8501
CMD [ "streamlit", "run", "demo_1_backup_feedback.py" ]