FROM python:3.10
RUN apt-get update && apt-get install python3 python3-pip git -y \
curl
RUN pip install --upgrade pip

CMD /bin/bash

COPY ./ ./
RUN pip install --no-cache-dir -r requirements.txt