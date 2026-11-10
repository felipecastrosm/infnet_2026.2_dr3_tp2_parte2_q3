from contextlib import asynccontextmanager

from fastapi import FastAPI

from routes import router
from services import get_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_chain()
    yield


app = FastAPI(title="HuggingFace Translation API with LangChain", version="1.0.0", lifespan=lifespan)

app.include_router(router)
