from fastapi import FastAPI
from backend.api import app as heartbeat_app

app = FastAPI()
app.mount('/api', heartbeat_app)
