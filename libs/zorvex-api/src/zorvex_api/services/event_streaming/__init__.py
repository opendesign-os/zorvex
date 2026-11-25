"""Agent Protocol v2 event streaming: a thread's run events over SSE and WebSocket, and its commands.

It follows the same run broker as the SDK's run streams (``/runs/stream``),
projecting each run's native v3 events onto the channels a client subscribes to.
"""
