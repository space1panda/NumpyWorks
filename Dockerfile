FROM python:3.6

WORKDIR /srv
ADD requirements.txt /srv
RUN pip install -r requirements.txt
ADD . /srv

CMD ["python", "cont_interface.py"]

