"""FastAPI application entrypoint."""
from fastapi import FastAPI

from api.routes import router
from observability.logger import configure_logging

configure_logging()

app = FastAPI(
    title="Arabic Customer Support Agent",
    description="Triage agent for Arabic-language customer support messages.",
    version="1.0.0",
)

app.include_router(router)
