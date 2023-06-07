FROM python:3.7.16-buster

# Make APT non-interactive
RUN echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/99semaphore
RUN echo 'DPkg::Options "--force-confnew";' >> /etc/apt/apt.conf.d/99semaphore
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt update && apt upgrade
RUN echo 'Acquire::Check-Valid-Until no;' >> /etc/apt/apt.conf
RUN apt-get install -y -m \
    git \
    mercurial \
    xvfb \
    vim \
    apt \
    locales \
    sudo \
    apt-transport-https \
    ca-certificates \
    openssh-client \
    software-properties-common \
    build-essential \
    tar \
    lsb-release \
    gzip \
    parallel \
    net-tools \
    netcat \
    unzip \
    zip \
    bzip2 \
    lftp \
    gnupg \
    curl \
    wget \
    build-essential \
    tree
RUN ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN locale-gen C.UTF-8 || true

RUN pip install --upgrade pip setuptools
RUN mkdir -p /var/lib/model
WORKDIR /var/lib/model

COPY . /var/lib/model

RUN pip3 install -r trial_1/requirements.txt