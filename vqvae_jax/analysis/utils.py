"""Shared utilities for VQ-VAE analysis modules.

Provides:
- ``identify_null_code``: Find the most frequent (null) code.
- ``CodeRun`` / ``extract_code_runs``: Extract contiguous code blocks.
- ``build_slider_html``: HTML slider viewer for videos or images.
"""

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .inference_cache import InferenceResult


def identify_null_code(results: Sequence[InferenceResult]) -> int:
    """Identify the null (most frequent) code across all results.

    Args:
        results: Inference results with code_indices populated.

    Returns:
        Index of the most frequent code.
    """
    all_codes = np.concatenate([r.code_indices for r in results])
    return int(np.argmax(np.bincount(all_codes)))


@dataclass
class CodeRun:
    """A contiguous block of the same code within a clip."""

    code: int
    start_frame: int
    end_frame: int  # exclusive


def extract_code_runs(code_indices: np.ndarray) -> list[CodeRun]:
    """Extract contiguous code runs from a sequence of code indices.

    Args:
        code_indices: Array of shape [T] with discrete code per frame.

    Returns:
        List of CodeRun objects representing contiguous blocks.
    """
    if len(code_indices) == 0:
        return []

    runs = []
    current_code = int(code_indices[0])
    start = 0

    for i in range(1, len(code_indices)):
        if int(code_indices[i]) != current_code:
            runs.append(CodeRun(code=current_code, start_frame=start, end_frame=i))
            current_code = int(code_indices[i])
            start = i

    # Final run
    runs.append(
        CodeRun(code=current_code, start_frame=start, end_frame=len(code_indices))
    )

    return runs


# =============================================================================
# HTML SLIDER VIEWER
# =============================================================================

_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def build_slider_html(
    media_paths: list[str],
    labels: list[str],
    title: str,
    media_type: str = "video",
) -> str:
    """Build an HTML page with a slider to browse videos or images.

    Each media item is embedded as base64. The slider selects which item to
    show.

    Args:
        media_paths: List of file paths (video or image).
        labels: Label for each item (shown below slider).
        title: Page title.
        media_type: ``"video"`` (default) embeds ``<video>`` elements;
            ``"image"`` embeds ``<img>`` elements.

    Returns:
        HTML string.
    """
    data_list: list[str] = []
    mime_list: list[str] = []
    for path in media_paths:
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            data_list.append(b64)
        except Exception:
            data_list.append("")

        if media_type == "image":
            ext = Path(path).suffix.lower()
            mime_list.append(_EXT_TO_MIME.get(ext, "image/png"))
        else:
            mime_list.append("video/mp4")

    labels_json = json.dumps(labels)
    mimes_json = json.dumps(mime_list)

    data_js = "[\n"
    for i, b64 in enumerate(data_list):
        data_js += f'  "{b64}"'
        if i < len(data_list) - 1:
            data_js += ","
        data_js += "\n"
    data_js += "]"

    if media_type == "image":
        media_el = '<img id="media" style="max-width:100%; border:1px solid #444;">'
        update_js = (
            "var m = document.getElementById('media');\n"
            "  m.src = 'data:' + mimes[i] + ';base64,' + items[i];"
        )
    else:
        media_el = '<video id="media" autoplay loop muted style="max-width:100%; border:1px solid #444;"></video>'
        update_js = (
            "var m = document.getElementById('media');\n"
            "  m.src = 'data:' + mimes[i] + ';base64,' + items[i];"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; text-align: center; background: #fff; color: #222; }}
  .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
  input[type=range] {{ width: 80%; margin: 15px 0; }}
  .label {{ font-size: 16px; margin: 10px 0; }}
</style>
</head>
<body>
<div class="container">
  <h2>{title}</h2>
  {media_el}
  <br>
  <input type="range" id="slider" min="0" max="{len(data_list) - 1}" value="0">
  <div class="label" id="lbl"></div>
</div>
<script>
var labels = {labels_json};
var mimes = {mimes_json};
var items = {data_js};
var slider = document.getElementById('slider');
var lbl = document.getElementById('lbl');
function update() {{
  var i = parseInt(slider.value);
  {update_js}
  lbl.textContent = labels[i];
}}
slider.addEventListener('input', update);
update();
</script>
</body>
</html>"""
    return html
