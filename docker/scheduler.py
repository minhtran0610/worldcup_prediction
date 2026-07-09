"""WC 2026 match scheduler — runs wc2026 --next 60 min before each kickoff and sends to Telegram."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID: str = os.environ["TELEGRAM_CHAT_ID"]
ADVANCE_MINUTES: int = int(os.getenv("ADVANCE_MINUTES", "60"))
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
WC2026_CMD: list[str] = [
    "wc2026",
    "--next",
    "--show-all",
    "--telegram",
    "--checkpoint",
    "checkpoints/neural_v2_quarterfinal.pt",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

_TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
_MAX_CHARS = 4096


def _send_chunk(text: str) -> None:
    resp = requests.post(
        _TELEGRAM_API,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        },
        timeout=15,
    )
    resp.raise_for_status()


def send_telegram(text: str) -> None:
    """Send text to Telegram, splitting at the API's 4096-char limit."""
    chunks = [text[i : i + _MAX_CHARS] for i in range(0, len(text), _MAX_CHARS)]
    for chunk in chunks:
        try:
            _send_chunk(chunk)
        except Exception as exc:
            log.error("Telegram send failed: %s", exc)


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------


def fire_prediction(home: str, away: str) -> None:
    """Run wc2026 --next, format the output, and send to Telegram."""
    log.info("Firing prediction for %s vs %s", home, away)

    env = {**os.environ, "OLLAMA_HOST": OLLAMA_HOST}
    try:
        result = subprocess.run(
            WC2026_CMD,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        output = "Prediction timed out after 10 minutes."
    except Exception as exc:
        output = f"Prediction error: {exc}"

    # Output is already Telegram HTML from --telegram flag; send each match block separately
    blocks = output.split("\n\n" + "—" * 32 + "\n\n")
    for block in blocks:
        send_telegram(block.strip())
    log.info("Report sent for %s vs %s", home, away)


# ---------------------------------------------------------------------------
# Schedule management
# ---------------------------------------------------------------------------


def schedule_matches(scheduler: BlockingScheduler) -> None:
    """Read the WC 2026 schedule and register a prediction job per upcoming match."""
    sys.path.insert(0, "/app")
    from data.ingest.wc2026 import load_wc2026_schedule

    log.info("Loading WC 2026 schedule...")
    try:
        schedule = load_wc2026_schedule(force_refresh=True)
    except Exception as exc:
        log.error("Failed to load schedule: %s", exc)
        return

    now = datetime.now(tz=UTC)
    registered = 0

    for row in schedule.itertuples(index=False):
        if getattr(row, "is_completed", False):
            continue

        kickoff: datetime = row.date
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=UTC)

        fire_at = kickoff - timedelta(minutes=ADVANCE_MINUTES)
        if fire_at <= now:
            continue

        job_id = f"{row.home_team}__vs__{row.away_team}"
        scheduler.add_job(
            fire_prediction,
            trigger=DateTrigger(run_date=fire_at, timezone=UTC),
            args=[row.home_team, row.away_team],
            id=job_id,
            replace_existing=True,
        )
        log.info(
            "Scheduled: %s vs %s — firing at %s UTC",
            row.home_team,
            row.away_team,
            fire_at.strftime("%Y-%m-%d %H:%M"),
        )
        registered += 1

    log.info("Registered %d upcoming match job(s).", registered)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    scheduler = BlockingScheduler(timezone="UTC")

    # Initial schedule load
    schedule_matches(scheduler)

    # Refresh daily at 02:00 UTC to pick up any FIFA schedule changes
    scheduler.add_job(
        schedule_matches,
        trigger=CronTrigger(hour=2, minute=0, timezone=UTC),
        args=[scheduler],
        id="daily_refresh",
        replace_existing=True,
    )

    log.info("Scheduler started. Waiting for match jobs...")
    scheduler.start()


if __name__ == "__main__":
    main()
