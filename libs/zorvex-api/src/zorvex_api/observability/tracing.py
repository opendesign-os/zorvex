"""OpenTelemetry tracing of graph runs, configured with the standard OTEL_* variables.

OTEL_TRACES_EXPORTER lists the exporters: otlp, console, or none (the default),
comma separated. The OTLP exporter (http/protobuf) reads OTEL_EXPORTER_OTLP_*
itself, so the endpoint, headers, timeout, compression and certificates, and
their *_TRACES_* variants, behave as the OpenTelemetry spec says. The resource
takes OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES; service.name defaults to
"zorvex". The SDK's span limits (OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, 128 by default,
and OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT) bound what a run's metadata adds.

A tracing backend such as Langfuse or Phoenix is an OTLP endpoint plus a header,
for example OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
with OTEL_EXPORTER_OTLP_TRACES_HEADERS=Authorization=Basic%20<base64 of public_key:secret_key>.

OpenTelemetry's FastAPI instrumentation makes each request a SERVER span, continuing
the trace context the request carries (W3C Trace Context and Baggage, or what
OTEL_PROPAGATORS names; OTEL_PYTHON_FASTAPI_EXCLUDED_URLS leaves routes out), and
OpenInference's LangChain instrumentation makes the graph's spans. As on the Agent
Server, a run created by a request that continues its caller's trace continues it
too: the run's configurable carries the request span's context to the worker
(__otel_traceparent__, __otel_tracestate__), which runs the graph under a CONSUMER
span, worker.stream_run, and its webhook carries the current span's context on.
Any other run is a trace of its own (its run id as the trace id, see
OTEL_TRACE_ID_FROM_RUN_ID), so a trace is still found by its run id.
Every span of a run carries the run's user and thread under the names
Phoenix reads (user.id, session.id) and those Langfuse filters on (langfuse.*),
with the run's metadata as langfuse.trace.metadata.<key>: Langfuse filters only
on what every span of a trace carries.
"""

import contextvars
import json
import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager

import structlog
from fastapi import FastAPI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.environment_variables import OTEL_TRACES_EXPORTER
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.environment_variables import OTEL_TRACES_SAMPLER
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import AttributeValue

from zorvex_api import __version__
from zorvex_api.models.run_job import RunJob
from zorvex_api.settings import settings
from zorvex_api.utils.run_utils import OTEL_TRACEPARENT, OTEL_TRACESTATE

logger = structlog.getLogger(__name__)

_EXPORTERS: dict[str, type[SpanExporter]] = {"otlp": OTLPSpanExporter, "console": ConsoleSpanExporter}

# Samplers that keep a trace by the bits of its id, which a run id's UUID biases.
_RATIO_SAMPLERS = frozenset({"traceidratio", "parentbased_traceidratio"})

# What the spans of the run executing in this task carry, and the trace id they share.
_run_attributes: contextvars.ContextVar[dict[str, AttributeValue] | None] = contextvars.ContextVar(
    "zorvex_run_span_attributes", default=None
)
_run_trace_id: contextvars.ContextVar[int | None] = contextvars.ContextVar("zorvex_run_trace_id", default=None)

_tracing = False
_trace_id_from_run_id = False


class _RunIdGenerator(RandomIdGenerator):
    """The run id as the trace id while a run executes, so a trace is found by its run id."""

    def generate_trace_id(self) -> int:
        return _run_trace_id.get() or super().generate_trace_id()


class _RunAttributes(SpanProcessor):
    """Puts the executing run's attributes on every span it creates."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if attributes := _run_attributes.get():
            span.set_attributes(attributes)


def _requested_exporters() -> list[str]:
    """Exporter names from OTEL_TRACES_EXPORTER; "none" and unsupported names select nothing."""
    names: list[str] = []
    for raw in os.getenv(OTEL_TRACES_EXPORTER, "none").split(","):
        name = raw.strip().lower()
        if name in _EXPORTERS:
            names.append(name)
        elif name and name != "none":
            logger.warning("Ignoring an unsupported OTEL_TRACES_EXPORTER entry", entry=name, supported="otlp, console")
    return names


def _resource() -> Resource:
    """The SDK's resource from the environment, with Zorvex's version and a default service name."""
    resource = Resource.create({SERVICE_VERSION: __version__})
    if str(resource.attributes.get(SERVICE_NAME, "")).startswith("unknown_service"):
        resource = resource.merge(Resource({SERVICE_NAME: "zorvex"}))
    return resource


def setup_tracing(app: FastAPI) -> None:
    """Install the tracer provider for the exporters OTEL_TRACES_EXPORTER names, and instrument LangChain and *app*.

    *app* must not have served yet: its instrumentation wraps its middleware.
    Nothing happens when trace export is off; ``shutdown_tracing`` exports the
    spans still batched at exit.
    """
    global _tracing, _trace_id_from_run_id
    exporters = _requested_exporters()
    if not exporters:
        logger.info("Trace export is off (OTEL_TRACES_EXPORTER unset or none)")
        return

    _trace_id_from_run_id = settings.observability.OTEL_TRACE_ID_FROM_RUN_ID
    if _trace_id_from_run_id and os.getenv(OTEL_TRACES_SAMPLER, "").lower() in _RATIO_SAMPLERS:
        logger.warning(
            "OTEL_TRACE_ID_FROM_RUN_ID is off: a ratio sampler would sample by the biased bits of a run id's UUID"
        )
        _trace_id_from_run_id = False

    provider = TracerProvider(resource=_resource(), id_generator=_RunIdGenerator())
    provider.add_span_processor(_RunAttributes())
    for name in exporters:
        provider.add_span_processor(BatchSpanProcessor(_EXPORTERS[name]()))
        logger.info("Trace export enabled", exporter=name)
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    _tracing = True


def shutdown_tracing() -> None:
    """Export the spans still batched and stop the exporters; nothing when trace export is off."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()


# W3C Trace Context, as the Agent Server hands a run's trace to its worker and to its webhook.
_W3C = TraceContextTextMapPropagator()


def trace_headers() -> dict[str, str]:
    """The current span's W3C context (traceparent, tracestate); none while tracing is off or no span is current."""
    if not _tracing or not trace.get_current_span().get_span_context().is_valid:
        return {}
    carrier: dict[str, str] = {}
    _W3C.inject(carrier)
    return carrier


def trace_markers() -> dict[str, str]:
    """The request span's W3C context as a run's configurable carries it to its worker, when the request
    continues its caller's trace; a request that starts its own leaves the run a trace of its own."""
    parent = getattr(trace.get_current_span(), "parent", None)
    if parent is None or not parent.is_remote:
        return {}
    headers = trace_headers()
    markers = {OTEL_TRACEPARENT: headers["traceparent"]} if "traceparent" in headers else {}
    if "tracestate" in headers:
        markers[OTEL_TRACESTATE] = headers["tracestate"]
    return markers


@contextmanager
def run_span(job: RunJob) -> Generator[None]:
    """The run under a CONSUMER span, worker.stream_run, continuing the trace of the request that created it,
    when that request carried one and tracing is on; otherwise nothing."""
    configurable = job.kwargs.config.get("configurable") or {}
    traceparent = configurable.get(OTEL_TRACEPARENT)
    if not _tracing or not traceparent:
        yield
        return
    carrier = {"traceparent": traceparent}
    if tracestate := configurable.get(OTEL_TRACESTATE):
        carrier["tracestate"] = tracestate
    tracer = trace.get_tracer("zorvex", __version__)
    with tracer.start_as_current_span("worker.stream_run", context=_W3C.extract(carrier), kind=trace.SpanKind.CONSUMER):
        yield


def start_run_trace(job: RunJob) -> None:
    """Make the spans this task creates from now on one trace for the run, each carrying its attributes.

    The task that executes the run calls it before the graph starts. The
    caller's metadata yields to the run's own keys; values that are not
    scalars go as JSON, since Langfuse metadata values are strings.
    """
    if not _tracing:
        return
    # A trace of its own, not a child of whatever span the task inherited (run_span continues the caller's).
    # The context dies with the task.
    otel_context.attach(trace.set_span_in_context(trace.INVALID_SPAN))
    identity, user_id = job.identity, job.user.identity
    if _trace_id_from_run_id:
        _run_trace_id.set(uuid.UUID(identity.run_id).int)

    own = {"run_id": identity.run_id, "thread_id": identity.thread_id, "graph_id": identity.graph_id}
    if job.correlation_id:
        own["request_id"] = job.correlation_id
    attributes: dict[str, AttributeValue] = {
        "user.id": user_id,
        "langfuse.user.id": user_id,
        "session.id": identity.thread_id,
        "langfuse.session.id": identity.thread_id,
        "langfuse.trace.name": identity.graph_id,
    }
    for key, value in {**job.run_metadata, **own}.items():
        scalar = isinstance(value, str | int | float | bool)
        attributes[f"langfuse.trace.metadata.{key}"] = value if scalar else json.dumps(value, default=str)
    _run_attributes.set(attributes)
