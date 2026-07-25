"""cv2.imread/imwrite 在 Windows 上遇到路徑含非 ASCII 字元（本專案資料夾名稱
本身就是中文）會靜默失敗——它們底層用系統 ANSI codepage 呼叫 fopen。
這裡改用 np.fromfile/tofile + cv2.imdecode/imencode 繞過，全專案讀寫影像
一律走這兩個函式，不要直接用 cv2.imread/imwrite。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"無法讀取影像：{path}")
    return image


def imwrite_unicode(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"無法編碼影像：{path}")
    buf.tofile(str(path))
