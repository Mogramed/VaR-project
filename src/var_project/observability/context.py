from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator


_REQUEST_ID: ContextVar[str | None] = ContextVar("var_project_request_id", default=None)
_RUN_ID: ContextVar[str | None] = ContextVar("var_project_run_id", default=None)
_ACCOUNT_ID: ContextVar[str | None] = ContextVar("var_project_account_id", default=None)
_ACTION: ContextVar[str | None] = ContextVar("var_project_action", default=None)


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def get_run_id() -> str | None:
    return _RUN_ID.get()


def get_account_id() -> str | None:
    return _ACCOUNT_ID.get()


def get_action() -> str | None:
    return _ACTION.get()


def current_correlation_context() -> dict[str, str | None]:
    return {
        "request_id": get_request_id(),
        "run_id": get_run_id(),
        "account_id": get_account_id(),
        "action": get_action(),
    }


def _bind(tokens: list[tuple[ContextVar[str | None], Token[str | None]]], variable: ContextVar[str | None], value: object | None) -> None:
    if value in {None, "", "null"}:
        return
    tokens.append((variable, variable.set(str(value))))


@contextmanager
def bind_correlation_context(
    *,
    request_id: object | None = None,
    run_id: object | None = None,
    account_id: object | None = None,
    action: object | None = None,
) -> Iterator[None]:
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    _bind(tokens, _REQUEST_ID, request_id)
    _bind(tokens, _RUN_ID, run_id)
    _bind(tokens, _ACCOUNT_ID, account_id)
    _bind(tokens, _ACTION, action)
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)

