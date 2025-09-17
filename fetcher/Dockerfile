FROM alpine:latest
RUN echo "@edge https://dl-cdn.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN apk --no-cache add python3 py3-pydantic py3-rich py3-yaml py3-httpx py3-typer nfpm
COPY fetcher.py /usr/local/bin/fetcher
RUN chmod +x /usr/local/bin/fetcher

WORKDIR /app

ENTRYPOINT ["/usr/local/bin/fetcher"]