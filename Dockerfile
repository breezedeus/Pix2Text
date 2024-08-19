# FROM --platform=linux/arm64 python:3.9
FROM --platform=linux/amd64 python:3.9
ENV TZ=America/Los Angeles
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3-opencv libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

RUN  pip install -U pip && pip install onnxruntime && pip install pix2text

RUN pip install pix2text

CMD ["p2t", "serve", "-l", "en", "-H", "0.0.0.0", "-p", "8503"]

EXPOSE 8503
