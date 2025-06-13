import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Text(BaseModel):
    text: str

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def root():
    return {'greeting':'Hello'}

@app.post("/text")
def receive_text(my_text: Text):
    import ipdb; ipdb.set_trace()
    body = my_text.text
    num_words = len(body)
    return {"text_length": num_words}
