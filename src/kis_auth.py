"""Read-only KIS WebSocket authorization from server-side secrets."""
from __future__ import annotations

import os

import requests


def websocket_approval_key() -> str:
    """Use an explicit approval key or request a short-lived one from KIS."""
    approval = os.getenv("KIS_WS_APPROVAL_KEY", "").strip()
    if approval:
        return approval
    app_key = os.getenv("KIS_APP_KEY", "").strip()
    app_secret = os.getenv("KIS_APP_SECRET", "").strip()
    if not app_key or not app_secret:
        raise RuntimeError("Set KIS_WS_APPROVAL_KEY or both KIS_APP_KEY and KIS_APP_SECRET")
    try:
        response = requests.post(
            "https://openapi.koreainvestment.com:9443/oauth2/Approval",
            json={"grant_type": "client_credentials", "appkey": app_key, "secretkey": app_secret},
            timeout=(5, 15),
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise RuntimeError(f"KIS WebSocket authorization failed (HTTP {response.status_code})")
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError("KIS WebSocket authorization returned no usable response") from None
    approval = payload.get("approval_key") if isinstance(payload, dict) else None
    if not isinstance(approval, str) or not approval.strip():
        raise RuntimeError("KIS WebSocket authorization returned no approval key")
    return approval.strip()
