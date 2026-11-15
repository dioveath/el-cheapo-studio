from contextlib import contextmanager

import pytest

from studio import db


def capture_connection(monkeypatch):
    captured = {}

    @contextmanager
    def connect(**options):
        captured.update(options)
        yield captured

    monkeypatch.setattr(db.psycopg, "connect", connect)
    return captured


def test_password_is_passed_without_url_interpolation(monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    password = "test-only:@/?#%$' space"
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    captured = capture_connection(monkeypatch)
    with db.connection() as conn:
        assert conn is captured
    assert captured["password"] == password
    assert captured["host"] == "localhost"
    assert captured["dbname"] == captured["user"] == "studio"
    assert "conninfo" not in captured


def test_database_url_remains_supported(monkeypatch):
    url = "postgresql://studio@localhost/studio"
    monkeypatch.setattr(db, "DATABASE_URL", url)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    captured = capture_connection(monkeypatch)
    with db.connection():
        pass
    assert captured["conninfo"] == url
    assert "password" not in captured


@pytest.mark.parametrize("password", [None, ""])
def test_missing_password_fails_before_connecting(monkeypatch, password):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    if password is not None:
        monkeypatch.setenv("POSTGRES_PASSWORD", password)
    captured = capture_connection(monkeypatch)
    with pytest.raises(RuntimeError, match="Set DATABASE_URL or POSTGRES_PASSWORD"):
        with db.connection():
            pass
    assert captured == {}
