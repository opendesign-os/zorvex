"""Standard Webhooks signatures for run-completion webhooks, made with the spec's own library.

Webhook URLs are caller-supplied, so a receiver needs a way to tell a real
callback from anyone who learned the URL. Every delivery carries ``webhook-id``;
with ``ZORVEX_WEBHOOK_SECRET`` set it also carries ``webhook-timestamp`` and
``webhook-signature`` (https://www.standardwebhooks.com), so stock verifiers work.
"""

from datetime import UTC, datetime
from functools import cache

import structlog
from standardwebhooks import Webhook

from zorvex_api.settings import WEBHOOK_SECRET_PREFIX, settings

logger = structlog.getLogger(__name__)


@cache
def _signer() -> Webhook | None:
    """The signer, or None when deliveries go out unsigned.

    A ``whsec_`` key is base64 behind the prefix, as Standard Webhooks issues
    them; any other key is used as raw bytes.
    """
    secret = settings.webhook.ZORVEX_WEBHOOK_SECRET
    raw = secret.get_secret_value() if secret else None
    if not raw:
        return None
    return Webhook(raw if raw.startswith(WEBHOOK_SECRET_PREFIX) else raw.encode())


@cache
def _warn_unsigned() -> None:
    logger.warning("Webhooks are going out unsigned; set ZORVEX_WEBHOOK_SECRET so receivers can verify them")


def signature_headers(message_id: str, body: bytes) -> dict[str, str]:
    """Headers for one delivery attempt of *body*.

    Pass the same *message_id* on every attempt so a receiver can deduplicate
    retries on it. Each attempt is signed with the time it is sent, so a late
    retry does not land outside the receiver's replay window.
    """
    headers = {"webhook-id": message_id}
    signer = _signer()
    if signer is None:
        _warn_unsigned()
        return headers
    now = datetime.now(UTC)
    headers["webhook-timestamp"] = str(int(now.timestamp()))
    headers["webhook-signature"] = signer.sign(message_id, now, body)
    return headers
