from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from src.indicators import add_indicators


class ObvTests(unittest.TestCase):
    def test_signed_volume_and_flat_close(self):
        frame = pd.DataFrame({
            "open": [10, 12, 11, 11],
            "high": [11, 13, 12, 12],
            "low": [9, 10, 10, 10],
            "close": [10, 12, 11, 11],
            "volume": [100, 200, 50, 80],
        })
        result = add_indicators(frame)
        self.assertEqual(result["obv"].tolist(), [0.0, 200.0, 150.0, 150.0])

    def test_obv_uses_no_future_rows(self):
        n = 45
        frame = pd.DataFrame({
            "open": np.arange(n, dtype=float) + 10,
            "high": np.arange(n, dtype=float) + 11,
            "low": np.arange(n, dtype=float) + 9,
            "close": np.arange(n, dtype=float) + 10,
            "volume": np.arange(n, dtype=float) + 100,
        })
        original = add_indicators(frame.iloc[:35].copy())
        extended = add_indicators(frame.copy()).iloc[:35]
        for column in ("obv", "obv_change_5d", "obv_change_20d"):
            np.testing.assert_allclose(
                original[column].to_numpy(),
                extended[column].to_numpy(),
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
