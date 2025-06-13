FROM python:3.10.6-buster

#Set working directory for docker
WORKDIR /prod

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements.txt requirements.txt

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

#Copy package
COPY sentiscope sentiscope

#Copy model directly
COPY models models

#Copy nltk data needed
COPY nltk_data /usr/share/nltk_data
ENV NLTK_DATA=/usr/share/nltk_data

CMD uvicorn sentiscope.api.fast:app --host 0.0.0.0 --port $PORT
