FROM ubuntu

MAINTAINER AI_SATURDAY

# Update the repository sources list
RUN apt-get update && apt-get -y upgrade

RUN pip3 install -r requirements.txt

