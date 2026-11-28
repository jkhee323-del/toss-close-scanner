"""CVD must come from signed execution data, never OHLCV estimates."""
import pandas as pd
import pytest

from src.cvd import compute_cvd


def test_cvd_accumulates_real_aggressor_side_and_sorts():
    df = pd.DataFrame({
        "timestamp": [
            "2026-11-17T09:00:02+09:00",
            "2026-11-17T09:00:00+09:00",
            "2026-11-17T09:00:01+09:00",
        ],
        "volume": [7, 10, 4],
        "aggressor_side": ["SELL", "BUY", "buy"],
    })
    result = compute_cvd(df)
    assert result["signed_volume"].tolist() == [10, 4, -7]
    assert result["cvd"].tolist() == [10, 14, 7]


def test_cvd_can_reset_at_exchange_session_boundary():
    df = pd.DataFrame({
        "timestamp": ["2026-11-16T15:25:00+09:00", "2026-11-17T09:00:00+09:00"],
        "volume": [3, 5],
        "aggressor_side": ["buy", "sell"],
    })
    result = compute_cvd(df, reset_each_session=True)
    assert result["cvd"].tolist() == [3, -5]
    assert result["session_date"].tolist() == ["2026-11-16", "2026-11-17"]


@pytest.mark.parametrize("data", [
    {"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [3]},
    {"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [3], "aggressor_side": ["unknown"]},
    {"timestamp": ["2026-11-17T09:00:00"], "volume": [3], "aggressor_side": ["buy"]},
    {"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [0], "aggressor_side": ["buy"]},
])
def test_cvd_rejects_ambiguous_or_missing_data(data):
    with pytest.raises(ValueError):
        compute_cvd(pd.DataFrame(data))


@pytest.mark.parametrize("volume", [float("inf"), float("-inf"), float("nan")])
def test_cvd_rejects_nonfinite_volume(volume):
    with pytest.raises(ValueError):
        compute_cvd(pd.DataFrame({"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [volume], "aggressor_side": ["buy"]}))


def test_divergence_uses_completed_adjacent_bars_only():
    from src.cvd import cvd_divergence

    trades = pd.DataFrame({
        "timestamp": ["2026-11-20T09:01:00+09:00", "2026-11-20T09:06:00+09:00",
                      "2026-11-20T09:16:00+09:00", "2026-11-20T09:21:00+09:00"],
        "volume": [1, 3, 2, 1], "aggressor_side": ["buy", "buy", "sell", "buy"],
        "price": [100, 99, 101, 100],
    })
    signals = cvd_divergence(trades.iloc[::-1])
    assert signals["divergence"].tolist() == ["bullish"]
    assert signals["bar_end"].dt.strftime("%H:%M").tolist() == ["09:10"]


def test_kis_execution_volume_checks_prevent_gaps_and_replays():
    from src.kis_trades import normalize_kis_krx_trades

    raw = pd.DataFrame({
        "MKSC_SHRN_ISCD": ["005930"] * 3, "BSOP_DATE": ["20261120"] * 3,
        "STCK_CNTG_HOUR": ["090001", "090001", "090002"],
        "CCLD_DVSN": ["1", "5", "1"], "STCK_PRPR": [100, 101, 102],
        "CNTG_VOL": [2, 3, 4], "ACML_VOL": [102, 105, 109],
    })
    assert compute_cvd(normalize_kis_krx_trades(raw))["cvd"].tolist() == [2, -1, 3]
    with pytest.raises(ValueError, match="누락·중복"):
        normalize_kis_krx_trades(raw.drop(1))
    with pytest.raises(ValueError, match="누락·중복"):
        normalize_kis_krx_trades(raw.iloc[[0, 0, 1, 2]])
    with pytest.raises(ValueError, match="방향"):
        normalize_kis_krx_trades(raw.assign(CCLD_DVSN="3"))


def test_kis_message_parser_preserves_execution_order_and_side():
    from src.kis_cvd_capture import parse_trades

    def payload(time, side):
        fields = [""] * 46
        for index, value in {0: "005930", 33: "20261125", 1: time, 2: "90000",
                             12: "3", 13: "103", 21: side}.items():
            fields[index] = value
        return "^".join(fields)

    two = "^".join((payload("100001", "1"), payload("100002", "5")))
    assert [record[-1] for record in parse_trades("0|H0STCNT0|2|" + two, "005930")] == ["1", "5"]
    with pytest.raises(ValueError, match="Incomplete"):
        parse_trades("0|H0STCNT0|1|" + two, "005930")
    with pytest.raises(ValueError, match="different ticker"):
        parse_trades("0|H0STCNT0|2|" + two, "000660")


def test_kis_capture_calculates_delta_from_session_messages(monkeypatch):
    import json
    from contextlib import nullcontext
    from unittest.mock import Mock

    from src import kis_cvd_capture

    def payload(time, volume, total, side):
        fields = [""] * 46
        for index, value in {0: "005930", 33: "20261125", 1: time, 2: "90000",
                             12: volume, 13: total, 21: side}.items():
            fields[index] = value
        return "^".join(fields)

    socket = Mock()
    socket.recv.side_effect = [
        '{"header":{"tr_id":"H0STCNT0"},"body":{"rt_cd":"0"}}',
        '{"header":{"tr_id":"PINGPONG"}}',
        "0|H0STCNT0|3|" + "^".join([
            payload("090000", "100", "100", "3"), payload("090001", "3", "103", "1"),
            payload("090002", "2", "105", "5"),
        ]),
        TimeoutError,
    ]
    monkeypatch.setattr(kis_cvd_capture, "connect", lambda *args, **kwargs: nullcontext(socket))
    captured = kis_cvd_capture.capture("005930", 2, "local-test")
    assert captured["cvd"].tolist() == [3, 1]
    assert captured["source"].unique().tolist() == ["KIS_KRX_H0STCNT0"]
    assert json.loads(socket.send.call_args.args[0])["body"]["input"]["tr_key"] == "005930"
    socket.pong.assert_called_once()


def test_saved_exchange_totals_detect_removed_and_replayed_executions():
    trades = pd.DataFrame({
        "timestamp": ["2026-11-25T09:00:01+09:00", "2026-11-25T09:00:02+09:00",
                      "2026-11-25T09:00:03+09:00", "2026-11-26T09:00:01+09:00"],
        "volume": [3, 2, 4, 7], "aggressor_side": ["buy", "sell", "buy", "sell"],
        "accumulated_volume": [103, 105, 109, 7],
    })
    assert compute_cvd(trades)["cvd"].tolist() == [3, 1, 5, -2]
    with pytest.raises(ValueError, match="Missing, duplicate"):
        compute_cvd(trades.drop(1))
    with pytest.raises(ValueError, match="Missing, duplicate"):
        compute_cvd(trades.iloc[[0, 1, 1, 2, 3]])
    with pytest.raises(ValueError, match="Invalid exchange"):
        compute_cvd(trades.assign(accumulated_volume=float("inf")))


@pytest.mark.parametrize("corruption", [None, "missing_totals", "gap", "replay", "reverse", "symbol", "source", "session", "price", "fraction"])
def test_collector_file_validation_survives_round_trip(tmp_path, corruption):
    from src.kis_trades import read_kis_capture

    captured = pd.DataFrame({
        "timestamp": ["2026-11-25T09:00:01+09:00", "2026-11-25T09:00:02+09:00", "2026-11-25T09:00:03+09:00"],
        "volume": [3, 2, 4], "accumulated_volume": [103, 105, 109],
        "aggressor_side": ["buy", "sell", "buy"], "price": [90000, 90100, 90200],
        "symbol": "005930", "source": "KIS_KRX_H0STCNT0",
    })
    if corruption == "missing_totals":
        captured = captured.drop(columns="accumulated_volume")
    elif corruption == "gap":
        captured = captured.drop(1)
    elif corruption == "replay":
        captured = captured.iloc[[0, 1, 1, 2]]
    elif corruption == "reverse":
        captured = captured.iloc[::-1]
    elif corruption == "symbol":
        captured.loc[1, "symbol"] = "000660"
    elif corruption == "source":
        captured["source"] = "OHLCV_estimate"
    elif corruption == "session":
        captured.loc[2, "timestamp"] = "2026-11-26T09:00:03+09:00"
    elif corruption == "price":
        captured = captured.assign(price=float("inf"))
    elif corruption == "fraction":
        captured["volume"] = captured["volume"].astype(float)
        captured.loc[0, "volume"] = 0.5
    path = tmp_path / "cvd-005930.csv"
    captured.to_csv(path, index=False)
    if corruption is not None:
        with pytest.raises(ValueError):
            read_kis_capture(path, "005930")
    else:
        result = read_kis_capture(path, "005930")
        assert result["cvd"].tolist() == [3, 1, 5]
        assert result["price"].tolist() == [90000, 90100, 90200]
        assert result["symbol"].unique().tolist() == ["005930"]
