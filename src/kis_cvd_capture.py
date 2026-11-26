"""Capture a bounded window of signed KRX executions from KIS H0STCNT0."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import pandas as pd
from websockets.sync.client import connect

from .cvd import compute_cvd
from .kis_auth import websocket_approval_key
from .kis_trades import KIS_EXECUTION_COLUMNS, normalize_kis_krx_trades

TR_ID = "H0STCNT0"
# Offsets within the provider's 46-field trade message, in CSV column order.
FIELD_OFFSETS = (0, 33, 1, 2, 12, 13, 21)


def parse_trades(message: str, ticker: str) -> list[tuple[str, ...]]:
    encoding, tr_id, count, payload = message.split("|", 3)
    if encoding != "0" or tr_id != TR_ID or not count.isdecimal():
        raise ValueError("Unexpected KIS execution message type")
    values = payload.split("^")
    if int(count) < 1 or len(values) != 46 * int(count):
        raise ValueError("Incomplete KIS execution message")
    records = [tuple(values[start + offset] for offset in FIELD_OFFSETS)
               for start in range(0, len(values), 46)]
    if any(record[0] != ticker for record in records):
        raise ValueError("KIS execution message contains a different ticker")
    return records


def capture(ticker: str, seconds: int, approval_key: str) -> pd.DataFrame:
    if not re.fullmatch(r"[0-9]{6}", ticker) or not 1 <= seconds <= 23400:
        raise ValueError("Provide a six-digit ticker and a capture duration between 1 and 23400 seconds")
    if not approval_key:
        raise RuntimeError("KIS_WS_APPROVAL_KEY is required")
    subscribe = {
        "header": {"approval_key": approval_key, "custtype": "P", "tr_type": "1", "content-type": "utf-8"},
        "body": {"input": {"tr_id": TR_ID, "tr_key": ticker}},
    }
    records = []
    with connect("ws://ops.koreainvestment.com:21000", ping_interval=None, open_timeout=10) as socket:
        socket.send(json.dumps(subscribe))
        deadline = monotonic() + seconds
        while (remaining := deadline - monotonic()) > 0:
            try:
                message = socket.recv(timeout=remaining, decode=True)
            except TimeoutError:
                break
            if message.startswith("{"):
                response = json.loads(message)
                message_type = response.get("header", {}).get("tr_id")
                if message_type == "PINGPONG":
                    socket.pong(message.encode())
                elif message_type != TR_ID or response.get("body", {}).get("rt_cd") != "0":
                    raise RuntimeError("KIS rejected the KRX execution subscription")
                continue
            for record in parse_trades(message, ticker):
                if not records and record[-1] == "3":
                    continue
                records.append(record)
                if len(records) > 1_000_000:
                    raise RuntimeError("KIS capture exceeded one million executions")
    if not records:
        raise RuntimeError("No signed KRX executions received")
    raw = pd.DataFrame.from_records(records, columns=KIS_EXECUTION_COLUMNS)
    trades = normalize_kis_krx_trades(raw)
    delta = compute_cvd(trades)
    return delta.assign(price=trades["price"], source="KIS_KRX_H0STCNT0", symbol=ticker,
                        accumulated_volume=pd.to_numeric(raw["ACML_VOL"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    delta = capture(args.ticker, args.seconds, websocket_approval_key())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as temp:
        pending = Path(temp.name)
    try:
        delta.to_csv(pending, index=False)
        pending.replace(args.output)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Captured and validated {len(delta)} signed KRX executions for {args.ticker}")


if __name__ == "__main__":
    main()
