FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
MAINTAINER caizebin
ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.8
RUN apt-get update -y
RUN apt-get install -y unzip wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-distutils python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
COPY . /app
WORKDIR app/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install notebook
