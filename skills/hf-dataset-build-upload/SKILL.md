---
name: hf-dataset-build-upload
description: Build a multi-split Hugging Face dataset with typed Features (including Image/Video/ClassLabel/Sequence columns), push it with push_to_hub, attach a dataset card, and resolve dataset asset URLs at read time without re-downloading (so providers can fetch them directly). Covers token handling and human-readable JSON mirrors.
---

# Building & Uploading Hugging Face Datasets

A reusable pattern for turning local records into a typed, multi-split HF dataset
and for *reading* assets back without redundant downloads. Domain-neutral.

## Declare typed Features per split

Don't rely on type inference — declare `Features` explicitly so columns get the
right dtype and media columns are handled by the Hub. One `Features` per split:

```python
from datasets import Dataset, Features, Value, Sequence, ClassLabel, Image

vqa_features = Features({
    "question_id": Value("string"), "video_id": Value("string"),
    "question": Value("string"),
    "choice_A": Value("string"), "choice_B": Value("string"),
    "answer": Value("string"),
})
cls_features = Features({                       # categorical label, not a string
    "video_id": Value("string"),
    "label": ClassLabel(names=["negative", "positive"]),
})
seg_features = Features({                        # media + variable-length lists
    "video_id": Value("string"),
    "first_mask": Image(),                       # decoded/rendered by the Hub
    "mask_frame_indices": Sequence(Value("int32")),
    "mask_paths": Sequence(Value("string")),
})

ds = Dataset.from_list(records, features=vqa_features)
```

Use `ClassLabel` for categoricals (enables stratification + int encoding),
`Image()`/`Video()` for media so the Hub renders previews, and `Sequence(...)`
for variable-length fields.

## Push splits + attach a dataset card

Pass the token directly to every call (don't depend on ambient login). Push each
split under its own config/split name, then upload `README.md` as the card:

```python
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

vqa_ds.push_to_hub(repo_id, config_name="vqa", token=HF_TOKEN)
cls_ds.push_to_hub(repo_id, config_name="classification", token=HF_TOKEN)

api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md",
                repo_id=repo_id, repo_type="dataset",
                commit_message="Add dataset card")
```

## Always mirror media-bearing splits as human-readable JSON

`push_to_hub` writes parquet, which is opaque to grep/diff. For any split you'll
inspect or regenerate, also dump a plain JSON next to it. This keeps the dataset
debuggable and re-buildable — the same philosophy as
[[json-checkpoint-records]]:

```python
json.dump(raw_records, open("raw.json", "w"), indent=2)
```

## Read-side: resolve asset URLs without re-downloading

When a downstream consumer (e.g. an API model) needs the *file*, resolve the
dataset `/resolve/...` path to its final fetchable CDN location and hand that URL
over directly — no local download, no re-upload to another bucket:

```python
from huggingface_hub import get_hf_file_metadata, hf_hub_url, hf_hub_download

def resolve_remote_url(repo_id, relative_path, token, revision="main"):
    url = hf_hub_url(repo_id=repo_id, filename=relative_path,
                     repo_type="dataset", revision=revision)
    return get_hf_file_metadata(url, token=token).location or url   # CDN location
```

Cache resolved URLs with a TTL (they expire). Keep a `download_video()` companion
that prefers an existing local copy, else `hf_hub_download` into the shared cache
— for consumers that *can't* fetch remote URLs (see [[mllm-video-input]] on which
providers need local bytes). This resolver is what powers the zero-upload video
path in [[interchangeable-model-backends]].

## Token resolution helper

Accept an explicit token, else fall back through the standard env names, else let
the library use the logged-in credentials:

```python
def get_hf_auth_token(explicit=None):
    tok = explicit or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return tok if tok else True   # True -> use local login
```
