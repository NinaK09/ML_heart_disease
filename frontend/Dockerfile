FROM python:3.9-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN export FLASK_APP=app

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]