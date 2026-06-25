---
name: mllm-detection
description: Use multimodal LLMs (Gemini and non-Gemini) for object detection / localization on images and video frames. Covers the prompt contract, the Gemini-vs-others coordinate convention split, and robust multi-fallback response parsing into pixel bounding boxes.
---

# MLLM-based Detection (Gemini vs non-Gemini)

How to make any vision LLM emit bounding boxes and turn its free-text answer
into reliable pixel coordinates. The hard part is **not** the API call — it is
(1) a prompt that forces a parseable format and (2) a parser that survives every
way the model deviates from it.

## 1. The single biggest gotcha: coordinate conventions differ by provider

Gemini returns box coordinates **normalized to a 0–1000 grid**, regardless of
the real image resolution. Almost every other provider (OpenAI, Qwen, GLM,
Molmo, …) returns coordinates already in **pixel space**. If you treat them the
same you get boxes that are either microscopic or far off-screen.

Branch on the provider **at parse time**, not at prompt time:

```python
def convert_bbox_to_pixels(raw_bbox, width, height, model_is_gemini):
    x1, y1, x2, y2 = raw_bbox
    # Gemini: 0-1000 normalized grid -> scale to real pixels.
    if model_is_gemini and max(raw_bbox) <= 1000.0:
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
    # clamp into frame, then fix inverted corners
    x1, x2 = sorted((clamp(round(x1), 0, width),  clamp(round(x2), 0, width)))
    y1, y2 = sorted((clamp(round(y1), 0, height), clamp(round(y2), 0, height)))
    return [x1, y1, x2, y2]
```

The `max(raw_bbox) <= 1000.0` guard is deliberate: some Gemini variants already
return pixels for large images, so only rescale when the values plausibly live
on the normalized grid.

## 2. Prompt contract: pin an exact, line-oriented output format

Don't ask for "the bounding box." Specify a delimiter-separated line format with
an explicit "nothing found" sentinel, and tell the model the coordinate space.

```
Task: detect any visible <target> and output bounding boxes.
Output format (one line per object):
FOUND | x_min,y_min,x_max,y_max | confidence(0-1) | short description
If nothing is visible:
NONE | nothing visible
Coordinates should be in image pixel coordinates.
```

Pipe-delimited lines beat JSON for small models: they fail to close brackets far
less often, and a partial line is still parseable. Always provide the empty
sentinel (`NONE`) so "no detection" is a first-class answer, not a parse error.

## 3. Parse defensively, with a fallback ladder

Models drift from any format. Parse in descending order of confidence and stop
at the first tier that yields boxes:

1. **Primary**: the line format above (`FOUND | ... | ... | ...`). Split on `|`,
   pull the first 4 numbers from the coords field with a regex
   `-?\d+(?:\.\d+)?`, parse confidence (accept `0-1` or `0-100` and divide).
2. **JSON fallback**: locate the outermost `[` … `]`, `json.loads` it, accept
   `bbox_2d` or `bbox` keys (Qwen-style emits `bbox_2d`).
3. **Last resort**: grab the first 4 numbers anywhere in the response and treat
   them as one box with confidence 1.0.

```python
def parse_detection_response(raw_text, width, height, model_is_gemini):
    text = (raw_text or "").strip()
    if not text or text.upper().startswith("NONE"):
        return []
    out = []
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        if not line.upper().startswith("FOUND"):
            continue
        parts = [p.strip() for p in line.split("|")]
        raw = parse_bbox_numbers(parts[1]) if len(parts) > 1 else None
        if raw is None:
            continue
        out.append({"bbox": convert_bbox_to_pixels(raw, width, height, model_is_gemini),
                    "confidence": parse_confidence(parts[2]) if len(parts) > 2 else 1.0,
                    "description": parts[3] if len(parts) > 3 else ""})
    if out:
        return out
    # ... JSON fallback, then first-4-numbers fallback ...
    return out
```

The principle: **never let one malformed response abort the run** — return `[]`
and let the run continue (see [[fault-tolerant-api-runs]]).

## 4. Frames vs native video for detection

For *per-frame* detection, send a single frame as an image part — even for
"video-capable" models — because you need box coordinates tied to one known
frame geometry (width/height). Reserve native-video input for holistic
questions over a clip (see [[mllm-video-input]]).

- **Gemini**: inline the frame as a content part
  `{"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}}`.
- **OpenRouter / others**: pass `frames_b64=[frame_b64], use_video=False`.

Both paths go through a unified completion wrapper that normalizes the response
shape (see [[provider-api-utils]] and [[interchangeable-model-backends]]).

## 5. Turning detections into downstream metrics

Detection output feeds standard metrics: greedy one-to-one IoU matching for
P/R/F1 at thresholds, and confidence-ranked AP with Pascal-style interpolation
(AP@50 and mAP@50-95). Keep the raw model text in the per-frame record so a
parser bug can be re-run later without re-calling the API — that re-runnability
is the whole point of [[json-checkpoint-records]].
