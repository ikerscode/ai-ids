"""Terminal HUD spec: every view answers, and every action points at a real route."""
import importlib

from fastapi.testclient import TestClient

from aiids import termhud
from aiids.tui_spec import SPEC


def client(tmp_data):
    import aiids.api as api
    importlib.reload(api)
    return TestClient(api.app), api


def test_views_return_200_and_rows_are_lists(tmp_data):
    c, _ = client(tmp_data)
    headers = {"host": "127.0.0.1"}
    for view in SPEC["views"]:
        r = c.get(view["endpoint"], headers=headers)
        assert r.status_code == 200, view["endpoint"]
        if "rows" in view:
            assert isinstance(termhud.dig(r.json(), view["rows"]), list), view["endpoint"]


def test_actions_point_at_existing_routes(tmp_data):
    _, api = client(tmp_data)
    routes = {(m, getattr(r, "path", None)) for r in api.app.routes for m in getattr(r, "methods", set())}
    for action in SPEC["actions"]:
        assert len(action["key"]) == 1 and action["key"] not in "q0123456789", action
        method = action.get("method", "POST")
        assert (method, action["endpoint"]) in routes, action


def test_clean_strips_escape_sequences():
    out = termhud.clean("\x1b[2J" + "hello")
    assert "\x1b" not in out
    assert "\x1b[2J" not in out
