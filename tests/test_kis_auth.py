from unittest.mock import Mock

import pytest
import requests

from src import kis_auth


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.delenv("KIS_WS_APPROVAL_KEY", raising=False)
    monkeypatch.setenv("KIS_APP_KEY", "test-app")
    monkeypatch.setenv("KIS_APP_SECRET", "test-secret")


def test_explicit_approval_avoids_token_request(monkeypatch):
    monkeypatch.setenv("KIS_WS_APPROVAL_KEY", "current-key")
    post = Mock()
    monkeypatch.setattr(kis_auth.requests, "post", post)
    assert kis_auth.websocket_approval_key() == "current-key"
    post.assert_not_called()


def test_approval_request_uses_documented_secret_field(monkeypatch):
    post = Mock(return_value=Mock(status_code=200, json=lambda: {"approval_key": "issued"}))
    monkeypatch.setattr(kis_auth.requests, "post", post)
    assert kis_auth.websocket_approval_key() == "issued"
    assert post.call_args.kwargs["json"] == {
        "grant_type": "client_credentials", "appkey": "test-app", "secretkey": "test-secret",
    }
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("payload", [{}, {"approval_key": None}, {"approval_key": ""}, []])
def test_missing_approval_is_rejected(monkeypatch, payload):
    monkeypatch.setattr(kis_auth.requests, "post", Mock(return_value=Mock(status_code=200, json=lambda: payload)))
    with pytest.raises(RuntimeError, match="no approval key"):
        kis_auth.websocket_approval_key()


def test_auth_errors_do_not_echo_secrets(monkeypatch):
    monkeypatch.setattr(kis_auth.requests, "post", Mock(side_effect=requests.Timeout("test-secret")))
    with pytest.raises(RuntimeError) as error:
        kis_auth.websocket_approval_key()
    assert "test-secret" not in str(error.value)


def test_missing_credentials_fail_before_network(monkeypatch):
    monkeypatch.delenv("KIS_APP_SECRET")
    post = Mock()
    monkeypatch.setattr(kis_auth.requests, "post", post)
    with pytest.raises(RuntimeError, match="KIS_APP_SECRET"):
        kis_auth.websocket_approval_key()
    post.assert_not_called()


def test_redirect_and_http_error_responses_are_rejected(monkeypatch):
    response = Mock(status_code=302)
    monkeypatch.setattr(kis_auth.requests, "post", Mock(return_value=response))
    with pytest.raises(RuntimeError, match="HTTP 302"):
        kis_auth.websocket_approval_key()
    response.json.assert_not_called()
