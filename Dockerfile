FROM python:3.7-alpine

MAINTAINER AI_SATURDAY

# Update the repository sources list
# RUN apt-get update && apt-get -y upgrade

COPY . /App

WORKDIR /App

RUN pip3 install -r requirements.txt

