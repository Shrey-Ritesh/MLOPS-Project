"""The real middleware stack must build; handler-level tests do not exercise it."""
from src.api.app import create_app


def test_middleware_stack_builds():
    app = create_app()
    assert app.build_middleware_stack() is not None


def test_middleware_unpacks_to_cls_args_kwargs():
    app = create_app()
    for middleware in app.user_middleware:
        cls, args, kwargs = middleware
        assert callable(cls)
