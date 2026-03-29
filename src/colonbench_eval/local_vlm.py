from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2

torch = None
AutoProcessor = None
Qwen3VLForConditionalGeneration = None
process_vision_info = None


def _load_local_deps() -> None:
    global torch, AutoProcessor, Qwen3VLForConditionalGeneration, process_vision_info
    if torch is not None:
        return

    import torch as _torch
    from qwen_vl_utils import process_vision_info as _process_vision_info
    from transformers import AutoProcessor as _AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration

    torch = _torch
    AutoProcessor = _AutoProcessor
    Qwen3VLForConditionalGeneration = _Qwen3VLForConditionalGeneration
    process_vision_info = _process_vision_info


def _video_geometry(video_path: str) -> Tuple[int, int, int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return total_frames, width, height


class LocalQwenVideoVLM:
    """Minimal local Qwen3-VL wrapper for ColonBench VQA."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        nframes: int = 32,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
        max_new_tokens: int = 32,
        verbose: bool = False,
    ):
        self.model_id = model_id
        self.nframes = nframes
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.processor = None
        self.model = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        _load_local_deps()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )

    def answer(self, *, video_path: str, prompt_text: str) -> Tuple[str, Dict[str, int]]:
        self._ensure_model()
        total_frames, width, height = _video_geometry(video_path)

        video_content: Dict[str, Any] = {
            "type": "video",
            "video": video_path,
            "nframes": max(self.nframes, 1),
        }
        if self.resized_height is not None:
            video_content["resized_height"] = self.resized_height
        elif height > 0:
            video_content["resized_height"] = height
        if self.resized_width is not None:
            video_content["resized_width"] = self.resized_width
        elif width > 0:
            video_content["resized_width"] = width
        if total_frames > 0 and "nframes" not in video_content:
            video_content["nframes"] = total_frames

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            first_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(first_device, int):
                first_device = f"cuda:{first_device}"
        else:
            first_device = self.model.device
        inputs = inputs.to(first_device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion_tokens = int(generated_ids_trimmed[0].shape[-1])
        return output_text, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
