"""
backend/app/core/logging.py
============================
Structured logging for NexusIQ.

CONCEPT: Why structured logging matters for AI systems
-------------------------------------------------------
Standard Python logging outputs plain text like:
  "2024-01-15 10:23:45 - Retrieving documents"

Structured logging outputs JSON like:
  {
    "timestamp": "2024-01-15T10:23:45Z",
    "level": "INFO",
    "service": "retrieval_agent",
    "query": "what is our refund policy",
    "chunks_retrieved": 10,
    "latency_ms": 234,
    "user_id": "user_123"
  }

JSON logs can be:
- Queried in Grafana/Datadog/CloudWatch
- Filtered by user_id, agent, or latency
- Used to debug agent behavior at scale

For your interview: "I used structured JSON logging from day one,
which made debugging agent failures in production straightforward —
I could filter all logs for a specific user's failed query."

We use Loguru — much cleaner API than Python's logging module.
"""

import sys
from loguru import logger
from backend.app.core.config import get_settings


def setup_logging() -> None:
    """
    Configure application-wide logging.
    Call this once at application startup in main.py.
    """
    settings = get_settings()

    # Remove default Loguru handler
    logger.remove()

    # ─── Development: Pretty colored output ─────────────────
    if settings.debug:
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
    else:
        # ─── Production: Structured JSON output ─────────────
        # JSON logs are machine-readable — ingested by log aggregators
        logger.add(
            sys.stdout,
            level=settings.log_level,
            format="{time:YYYY-MM-DDTHH:mm:ssZ} | {level} | {name} | {message} | {extra}",
            serialize=True,  # Outputs as JSON
        )

    # ─── File logging (always on) ───────────────────────────
    # Keeps logs even if stdout is lost. Rotates daily, keeps 7 days.
    logger.add(
        "logs/nexusiq_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",    # New file every day
        retention="7 days",  # Keep last 7 days
        compression="zip",   # Compress old logs
        serialize=True,      # JSON format for log analysis
    )

    logger.info(
        "NexusIQ logging initialized",
        env=settings.app_env,
        level=settings.log_level,
    )


# ─── Agent-specific logger factory ──────────────────────────
def get_agent_logger(agent_name: str):
    """
    Creates a logger bound to a specific agent name.
    
    Usage:
        log = get_agent_logger("retrieval_agent")
        log.info("Retrieved chunks", count=10, latency_ms=234)
    
    All logs will include agent_name automatically — 
    crucial for debugging which agent caused an issue.
    """
    return logger.bind(agent=agent_name)
