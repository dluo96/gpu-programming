FROM ubuntu:latest

RUN apt-get update && \
    apt-get -y install gcc make zlib1g-dev gnupg zlib1g time && \
    apt-get -y clean && \
    rm -rf \
      /var/lib/apt/lists/* \
      /usr/share/doc \
      /usr/share/doc-base \
      /usr/share/man \
      /usr/share/locale \
      /usr/share/zoneinfo

WORKDIR /home/daniel  
COPY . .
RUN make compile
 