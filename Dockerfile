FROM python:3.12-slim

RUN apt-get update && apt-get install -y libqt5gui5

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT [ "/app/entrypoint.sh" ]