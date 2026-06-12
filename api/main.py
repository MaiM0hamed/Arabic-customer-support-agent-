"""FastAPI application entrypoint."""
import logging

from fastapi import FastAPI

from api.routes import router
from api.security import RateLimitMiddleware
from config import settings
from observability.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

if not settings.api_key:
    logger.warning("API_KEY is not configured; all endpoints are unauthenticated.")

app = FastAPI(
    title="Arabic Customer Support Agent",
    description="Triage agent for Arabic-language customer support messages.",
    version="1.0.0",
)

app.add_middleware(RateLimitMiddleware)
app.include_router(router)
