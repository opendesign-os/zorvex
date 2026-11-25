"""Webhook configuration and URL policy.

Reads the ``webhooks`` block of zorvex.json (the same keys LangGraph's
langgraph.json uses) and decides which user-supplied webhook URLs the server
agrees to call. The checks run when a run or cron is created and again right
before every call, so a policy change also covers webhooks stored earlier.
"""

import asyncio
import ipaddress
import socket

import httpx
import structlog
from fastapi import HTTPException
from pydantic import BaseModel

from zorvex_api.config import WebhookUrlPolicy, load_config

logger = structlog.getLogger(__name__)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

_DNS_TIMEOUT_SECONDS = 10.0

_LOCALHOST_NAMES = frozenset({"localhost", "localhost.localdomain", "host.docker.internal"})
_METADATA_HOSTNAMES = frozenset({"metadata", "metadata.google.internal", "metadata.amazonaws.com", "instance-data"})
_K8S_SERVICE_SUFFIX = ".svc.cluster.local"

_LOOPBACK_NETWORKS = tuple(ipaddress.ip_network(net) for net in ("127.0.0.0/8", "::1/128"))
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "fc00::/7")
)
# Link-local, "this network", multicast, reserved, benchmarking and documentation ranges.
_ALWAYS_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/8",
        "169.254.0.0/16",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/96",
        "64:ff9b:1::/48",
        "fe80::/10",
        "ff00::/8",
    )
)
# Some providers serve instance metadata from addresses inside the private ranges.
_METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(addr) for addr in ("169.254.169.254", "169.254.170.2", "100.100.100.200", "fd00:ec2::254")
)
_NAT64_NETWORK = ipaddress.IPv6Network("64:ff9b::/96")


class WebhookUrlError(ValueError):
    """A webhook URL the server refuses to call. The message is safe to return to the caller."""


class WebhookDNSError(WebhookUrlError):
    """The host did not resolve: a 422 when a webhook is set, a retry when it is called."""


def webhook_host(url: httpx.URL) -> str:
    """The ASCII host a request to *url* connects to, without a trailing dot."""
    return url.raw_host.decode("ascii").rstrip(".")


def webhook_port(url: httpx.URL) -> int:
    """The port a request to *url* connects to."""
    if url.port is not None:
        return url.port
    return 443 if url.scheme == "https" else 80


def check_webhook_url(url: str, policy: WebhookUrlPolicy) -> httpx.URL:
    """Apply the parts of *policy* that need no DNS lookup and return the parsed URL.

    Parsed with httpx, the client that later sends the request, so both agree on the host.
    """
    if len(url) > policy.max_url_length:
        raise WebhookUrlError("Webhook URL too long")
    if url.startswith("/"):
        raise WebhookUrlError("Relative webhook URLs are not supported; use an absolute http or https URL")
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        raise WebhookUrlError("Invalid webhook URL") from None
    if parsed.scheme not in ("http", "https"):
        raise WebhookUrlError("Webhook must use http or https")
    if policy.require_https and parsed.scheme != "https":
        raise WebhookUrlError("Webhook must use https")
    host = webhook_host(parsed)
    if not host:
        raise WebhookUrlError("Invalid webhook hostname")
    port = webhook_port(parsed)
    if not 0 < port < 65536:
        raise WebhookUrlError("Invalid webhook URL")
    if policy.allowed_ports and port not in policy.allowed_ports:
        raise WebhookUrlError(f"Webhook port {port} not allowed")
    if policy.allowed_domains and not policy.allows_domain(host):
        raise WebhookUrlError("Webhook domain not allowed")
    return parsed


def _embedded_ipv4(address: ipaddress.IPv6Address) -> tuple[ipaddress.IPv4Address, ...]:
    """IPv4 addresses an IPv6 address carries: IPv4-mapped, NAT64, 6to4, or both Teredo endpoints."""
    if address.ipv4_mapped is not None:
        return (address.ipv4_mapped,)
    if address in _NAT64_NETWORK:
        return (ipaddress.IPv4Address(address.packed[-4:]),)
    if address.sixtofour is not None:
        return (address.sixtofour,)
    if address.teredo is not None:
        return address.teredo
    return ()


def _blocked_reason(address: IPAddress, policy: WebhookUrlPolicy) -> str | None:
    """Why *policy* forbids connecting to *address*, or None when it allows it."""
    if isinstance(address, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(address)
        if embedded:
            reasons = (_blocked_reason(ipv4, policy) for ipv4 in embedded)
            return next((reason for reason in reasons if reason is not None), None)
    if any(address in net for net in _LOOPBACK_NETWORKS):
        return "loopback IP range" if policy.disable_loopback else None
    if address in _METADATA_ADDRESSES:
        return "cloud metadata endpoint"
    if any(address in net for net in _ALWAYS_BLOCKED_NETWORKS):
        return "blocked IP range"
    if policy.disable_private_ips and any(address in net for net in _PRIVATE_NETWORKS):
        return "private IP range"
    return None


def _raise_if_blocked(address: IPAddress, policy: WebhookUrlPolicy) -> None:
    reason = _blocked_reason(address, policy)
    if reason is not None:
        raise WebhookUrlError(f"Webhook host blocked: {reason}")


def _check_hostname(host: str, policy: WebhookUrlPolicy) -> None:
    """Names that point inside the deployment no matter what DNS says."""
    if policy.disable_loopback and host in _LOCALHOST_NAMES:
        raise WebhookUrlError("Webhook host blocked: localhost address")
    if host in _METADATA_HOSTNAMES:
        raise WebhookUrlError("Webhook host blocked: cloud metadata endpoint")
    if host.endswith(_K8S_SERVICE_SUFFIX):
        raise WebhookUrlError("Webhook host blocked: Kubernetes internal DNS")


async def _resolve(host: str, port: int) -> list[IPAddress]:
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(_DNS_TIMEOUT_SECONDS):
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, TimeoutError):
        raise WebhookDNSError("Webhook host blocked: DNS resolution failed") from None
    try:
        addresses: list[IPAddress] = [ipaddress.ip_address(info[4][0]) for info in infos]
    except ValueError:
        raise WebhookUrlError("Webhook host blocked: invalid IP address") from None
    if not addresses:
        raise WebhookDNSError("Webhook host blocked: DNS resolution failed")
    return addresses


async def resolve_webhook_addresses(host: str, port: int, policy: WebhookUrlPolicy) -> list[str] | None:
    """Check where *host* points and return every resolved address, in DNS order, to pin to.

    None means use the URL's host as-is: an IP literal, a trusted host, or an allowed
    localhost alias (Docker can map those outside the loopback range).
    """
    if policy.trusts(host):
        return None
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        _raise_if_blocked(literal, policy)
        return None
    _check_hostname(host, policy)
    if host in _LOCALHOST_NAMES:
        return None
    addresses = await _resolve(host, port)
    for address in addresses:
        _raise_if_blocked(address, policy)
    return list(dict.fromkeys(str(address) for address in addresses))


async def validate_webhook_url(url: str) -> None:
    """Raise WebhookUrlError unless the server's policy allows calling *url*."""
    policy = load_config().webhooks.url
    parsed = check_webhook_url(url, policy)
    await resolve_webhook_addresses(webhook_host(parsed), webhook_port(parsed), policy)


async def admit_webhook[R: BaseModel](request: R) -> R:
    """*request* as the server takes it, as the Agent Server does: with ``http.disable_webhooks`` its
    webhook is dropped unchecked, as if never sent; otherwise one the policy will not call is a 422."""
    webhook = getattr(request, "webhook", None)
    if not webhook:
        return request
    if load_config().http.disable_webhooks:
        logger.info("Webhooks are disabled, dropping the request's webhook")
        kept = {name: getattr(request, name) for name in request.model_fields_set - {"webhook"}}
        return type(request).model_construct(set(kept), **kept)
    try:
        await validate_webhook_url(webhook)
    except WebhookUrlError as exc:
        raise HTTPException(422, str(exc)) from None
    return request
