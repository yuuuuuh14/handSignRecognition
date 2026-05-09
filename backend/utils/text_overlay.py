"""Korean-capable text overlay renderer for OpenCV frames.

OpenCV's `cv2.putText` uses Hershey vector fonts and only renders ASCII glyphs;
Korean characters appear as `?`. We render text via PIL with a TrueType font
that has Korean coverage (Malgun Gothic on Windows, Apple SD Gothic Neo on
macOS, NanumGothic / Noto CJK on Linux) and composite the result back onto
the OpenCV BGR frame.

`KoreanTextRenderer` buffers `text(...)` calls and flushes them in a single
BGR↔RGB↔PIL roundtrip per frame via `flush(img)`, which keeps the per-frame
overhead independent of the number of text strings drawn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Font candidates — first existing path wins. Listed in OS-priority order.
FONT_CANDIDATES: list[Path] = [
    Path("C:/Windows/Fonts/malgun.ttf"),                        # Windows Malgun Gothic
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/gulim.ttc"),
    Path("C:/Windows/Fonts/batang.ttc"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),         # macOS
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),    # Linux Nanum
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]


def load_korean_font(size: int) -> ImageFont.ImageFont:
    """Return the first available Korean-capable TrueType font at the given size."""
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    print(
        "[warn] no Korean-capable font found; falling back to PIL default "
        "(Korean characters may render as boxes)",
        file=sys.stderr,
    )
    return ImageFont.load_default()


class KoreanTextRenderer:
    """Buffers text-draw calls; flushes them via a single PIL roundtrip per frame."""

    def __init__(self, default_size: int = 18, small_size: int = 14) -> None:
        self.font_default = load_korean_font(default_size)
        self.font_small = load_korean_font(small_size)
        self._calls: list[
            tuple[str, tuple[int, int], tuple[int, int, int], ImageFont.ImageFont]
        ] = []

    def text(
        self,
        text: str,
        org: tuple[int, int],
        color: tuple[int, int, int] = (255, 255, 255),
        small: bool = False,
    ) -> None:
        """Buffer a text-draw call. `color` is BGR per OpenCV convention."""
        font = self.font_small if small else self.font_default
        self._calls.append((text, org, color, font))

    def flush(self, img_bgr: np.ndarray) -> np.ndarray:
        """Render all buffered calls onto a copy of `img_bgr` and clear the buffer."""
        if not self._calls:
            return img_bgr
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for text, org, color_bgr, font in self._calls:
            fill_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text(org, text, font=font, fill=fill_rgb,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        self._calls.clear()
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
