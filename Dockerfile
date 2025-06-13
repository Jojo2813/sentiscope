FROM python:3.10.6-buster

WORKDIR /prod

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements.txt requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

COPY sentiscope sentiscope
COPY models models

#Copy nltk data needed
COPY nltk_data /usr/share/nltk_data
ENV NLTK_DATA=/usr/share/nltk_data

#COPY home/marcvicente/.pyenv/versions/sentiscope/bin/uvicorn usr/local/bin/uvicorn

CMD uvicorn sentiscope.api.fast:app --host 0.0.0.0 --port $PORT
