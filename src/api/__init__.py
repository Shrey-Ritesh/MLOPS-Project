"""FastAPI Application package for JEE Tutor Evaluation Service."""
from __future__ import annotations

import inspect

import fastapi.applications
import starlette.middleware
import starlette.routing

# Backwards compatibility shims:
# 1. Older FastAPI releases pass on_startup/on_shutdown to Starlette Router.__init__,
#    which Starlette 1.x has removed from its init kwargs.
# 2. Older FastAPI unpacks Starlette Middleware as (cls, options), whereas Starlette 1.x
#    yields (cls, args, kwargs). Applied only when the installed FastAPI really unpacks
#    two values; with current FastAPI the shim would break every request with
#    "not enough values to unpack (expected 3, got 2)".
_orig_router_init = starlette.routing.Router.__init__


def _safe_router_init(self, *args, **kwargs):
    on_startup = kwargs.pop("on_startup", None)
    on_shutdown = kwargs.pop("on_shutdown", None)
    _orig_router_init(self, *args, **kwargs)
    self.on_startup = list(on_startup or [])
    self.on_shutdown = list(on_shutdown or [])


starlette.routing.Router.__init__ = _safe_router_init
if "cls, options" in inspect.getsource(fastapi.applications.FastAPI.build_middleware_stack):
    starlette.middleware.Middleware.__iter__ = lambda self: iter((self.cls, self.kwargs))

from .app import create_app

__all__ = ["create_app"]
