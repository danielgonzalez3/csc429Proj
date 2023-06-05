FROM python:3.7.16-buster

RUN echo "deb http://security.debian.org/debian-security bullseye-security main contrib non-free" > /etc/apt/sources.list
RUN apt-get update

RUN pip install --upgrade pip setuptools
RUN mkdir -p /var/lib/model
WORKDIR /var/lib/model

COPY . /var/lib/model

RUN pip3 install -r trial_1/requirements.txt

CMD python3 trial_1/trial_1.py