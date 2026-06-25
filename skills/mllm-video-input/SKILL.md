---
name: mllm-video-input
description: Feed videos to multimodal LLMs reliably. Covers the four input strategies (Gemini File API upload+poll, inline base64, remote video_url, evenly-spaced frame extraction), duration-aware frame scaling, size caps, and reference-counted upload caching so the same clip is uploaded once and cleaned up.
---

# Sending Video to Multimodal LLMs

There is no single way to give a video to a vision LLM — providers diverge hard.
This skill is the decision tree plus the four implementations, all
domain-neutral and reusable.

## Strategy selection

Pick per (provider, file size, availability of a remote URL):

| Situation | Strategy |
|-----------|----------|
| Gemini, any size | **File API**: upload, poll until `ACTIVE`, then reference |
| Non-Gemini, video-capable, file already at a fetchable URL | **remote `video_url`** (no upload, cheapest) |
| Non-Gemini, video-capable, small local file (< cap) | **inline base64 `video_url`** |
| Non-Gemini, video-capable, large local file | **upload to object storage, send signed URL** |
| Any model, no native video support | **frame extraction** (universal fallback) |

Native Gemini models **cannot fetch remote URLs** — they need the File API or
inline bytes. Most OpenRouter video models *can* fetch a URL, which is the
preferred zero-upload path.

## 1. Gemini File API: upload → poll → bake-in

The upload is async; you must poll for `ACTIVE` before referencing the file, and
add a short bake-in delay because file propagation across the backend lags the
state flip.

```python
def upload_video_to_gemini(client, video_path, verbose=False):
    f = client.files.upload(file=video_path)
    waited, max_wait, interval = 0, 300, 5
    while waited < max_wait:
        if f.state.name == "ACTIVE":
            break
        if f.state.name == "FAILED":
            raise RuntimeError("Video processing failed")
        time.sleep(interval); waited += interval
        f = client.files.get(name=f.name)
    time.sleep(3)  # bake-in: propagation lags the ACTIVE flip
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"Video processing timed out: {f.state.name}")
    return f
```

Uploaded files **count against a storage quota** and survive crashes. Run a
startup sweep that deletes all leftover files from interrupted prior runs:

```python
def cleanup_all_files(client):
    for f in client.files.list():
        try: client.files.delete(name=f.name)
        except Exception: pass
```

## 2. Frame extraction with duration-aware scaling

The universal fallback. A fixed frame count under-samples long clips and
over-samples short ones. Scale with duration toward a target frames-per-minute,
clamped to a hard cap to bound cost/context:

```python
FRAMES_PER_MINUTE, MAX_FRAMES_CAP = 8, 128

def extract_video_frames(path, num_frames=16, jpeg_quality=85, auto_scale=True):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n = num_frames
    if auto_scale:
        scaled = int(FRAMES_PER_MINUTE * (total / fps) / 60.0)
        n = max(num_frames, min(scaled, MAX_FRAMES_CAP))  # num_frames is a floor
    idx = [int(i * (total - 1) / (n - 1)) for i in range(n)] if n < total else range(total)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frames.append(base64.b64encode(buf).decode())
    cap.release()
    return frames
```

`num_frames` is a **floor**, not a target — short clips still get full coverage,
long clips scale up. Frames become `image_url` content parts.

## 3. Inline base64 vs object-storage URL (size-gated)

Base64 inflates payload ~33%. Providers enforce request-size limits (e.g. a
20 MB raw cap, and some providers a JSON string-length cap → cap inline video
even lower, e.g. 12–14 MB). Above the cap, upload to object storage and send a
**short-lived signed URL**:

```python
blob.upload_from_filename(path, content_type=mime)
url = blob.generate_signed_url(version="v4",
        expiration=timedelta(minutes=30), method="GET")
# pass url as a remote video_url; delete the blob right after the request
```

Treat these objects as ephemeral: upload → use → delete immediately. Keep a
per-model `max_video_bytes` override since limits are provider-specific.

## 4. Reference-counted upload cache (don't upload the same clip twice)

When many questions hit the same video, upload once and share it. Use a
thread-safe cache with **reference counting** so the file is deleted only when
the last consumer releases it — bounding storage even under concurrency:

```python
class FileCache:
    def get_or_upload(self, path):   # increments refcount; per-path lock
        ...                          # so concurrent callers wait, not re-upload
    def release(self, path):         # decrements; deletes from provider at 0
        ...
```

Contract: every successful `get_or_upload` **must** be paired with a `release`
(even on downstream failure); if the upload itself throws, no refcount is taken
and the caller must not release. Frame extraction uses the same caching idea
without the deletion/refcount, since frames are just in-memory bytes.

Related: provider wrappers in [[provider-api-utils]]; routing across providers
in [[interchangeable-model-backends]]; resilience in [[fault-tolerant-api-runs]].
