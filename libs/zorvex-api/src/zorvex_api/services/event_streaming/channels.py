"""The Agent Protocol v2 channels a client may subscribe to, as ``langchain_protocol.Channel`` defines them.

A channel is a top-level event stream (the ``method`` of a server event): the
protocol's named channels, and ``custom:<name>`` for user-defined custom events.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, get_args, get_origin

from langchain_protocol import Channel

# The named channels, and the patterns open-ended ones match (``custom:.+``).
SUPPORTED_CHANNELS: frozenset[str] = frozenset(
    get_args(member)[0] for member in get_args(Channel) if get_origin(member) is Literal
)
_PATTERNS = tuple(re.compile(get_args(member)[1]) for member in get_args(Channel) if get_origin(member) is Annotated)


def is_supported_channel(channel: str) -> bool:
    """True for one of the protocol's channels."""
    return channel in SUPPORTED_CHANNELS or any(pattern.fullmatch(channel) for pattern in _PATTERNS)
