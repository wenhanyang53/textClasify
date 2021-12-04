# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /app

RUN apt-get update

COPY . .

RUN pip3 install -r vocads/requirements.txt

CMD [ "python3", "prediction.py"]
