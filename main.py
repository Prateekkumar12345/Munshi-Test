

# main.py - FastAPI Backend
# Run: uvicorn main:app --reload

from contextlib import asynccontextmanager
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

from bot_engine import IntentClassifier, CommandParser

classifier = None
command_parser = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    global command_parser

    print("Loading classifier...")
    classifier = IntentClassifier()
    command_parser = CommandParser()
    print("Ready")

    yield

    print("Shutdown")


app = FastAPI(
    title="Worker Intent API",
    version="8.0.0",
    lifespan=lifespan,
)


class ClassifyResponse(BaseModel):
    intent: str
    id: Optional[Union[int, str]] = None
    worker_slug: Optional[str] = None
    depart_slug: Optional[str] = None
    deadline: Optional[str] = None
    message: Optional[str] = None


@app.get("/")
async def root():
    return {
        "service": "Hybrid Intent Classification API",
        "version": "8.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    message: str = Query(..., description="User message to classify")
):
    # Command parser first (highest priority)
    command_result = command_parser.parse(message)
    if command_result:
        return command_result

    # Hybrid classifier
    result = classifier.classify(message)
    return result


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )