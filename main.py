"""
main.py
FastAPI Backend

Run:
uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

from bot_engine import (
    IntentClassifier,
    CommandParser,
)

# ============================================================
# GLOBALS
# ============================================================

classifier = None

command_parser = None

# ============================================================
# LIFESPAN
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    global classifier
    global command_parser

    print("⏳ Loading Hybrid Intent Classifier...")

    classifier = IntentClassifier()

    command_parser = CommandParser()

    print("✅ API Ready")

    yield

    print("🛑 Shutdown")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Worker Intent Classification API",
    version="9.0.0",
    lifespan=lifespan,
)

# ============================================================
# RESPONSE MODEL
# ============================================================


class ClassifyResponse(BaseModel):

    intent: str

    id: Optional[Union[int, str]] = None

    worker_slug: Optional[str] = None

    depart_slug: Optional[str] = None

    deadline: Optional[str] = None

    message: Optional[str] = None


# ============================================================
# ROOT
# ============================================================


@app.get("/")
async def root():

    return {
        "service": "Hybrid Intent Classification API",
        "version": "9.0.0",
        "status": "running",
    }


# ============================================================
# HEALTH
# ============================================================


@app.get("/health")
async def health():

    return {
        "status": "ok",
    }


# ============================================================
# CLASSIFICATION ENDPOINT
# ============================================================


@app.post(
    "/classify",
    response_model=ClassifyResponse,
)
async def classify(
    message: str = Query(...)
):

    # ========================================================
    # COMMAND PARSER FIRST
    # ========================================================

    command_result = command_parser.parse(message)

    if command_result:
        return command_result

    # ========================================================
    # HYBRID CLASSIFIER
    # ========================================================

    result = classifier.classify(message)

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )