FROM ultralytics/ultralytics
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y \
curl
CMD /bin/bash

COPY ./ ./
RUN pip install --no-cache-dir -r requirements.txt