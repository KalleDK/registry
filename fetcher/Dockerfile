FROM debian:trixie-slim

RUN apt update -q -y
RUN apt install binutils ruby-rubygems --no-install-recommends -q -y
RUN gem install fpm

RUN apt install python3 python3-pydantic python3-httpx python3-typer --no-install-recommends -q -y

COPY ./fetcher.py /usr/local/bin/fetcher

RUN chmod +x /usr/local/bin/fetcher
WORKDIR /app

CMD ["fetcher"]