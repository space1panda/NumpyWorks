FROM python:3.6

WORKDIR /src
ADD requirements.txt /src
RUN pip install -r requirements.txt
ADD . /src
RUN mkdir tmp


ENTRYPOINT ["python", "cont_interface.py"]

