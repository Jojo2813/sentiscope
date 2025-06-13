FROM python:3.10.6-buster

WORKDIR /prod

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY api_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api api

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
