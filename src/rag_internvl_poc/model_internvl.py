from __future__ import annotations

import logging
import time
import os
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch

logger = logging.getLogger(__name__)

PROMPT_SYSTEM = (
    "Tu es un assistant qui répond de manière concise et factuelle en français en t'appuyant uniquement sur les extraits et images fournis en contexte. "
    "Si l'information n'est pas présente, indique-le."
)


def build_prompt(question: str, contexts: List[dict]) -> str:
    """Construit un prompt texte basique à partir des chunks récupérés.
    Chaque contexte inclut `content` (texte) et éventuellement un `image_path`.
    """
    parts = ["CONTEXTE:"]
    for i, c in enumerate(contexts, 1):
        header = f"[Contexte {i} | doc={os.path.basename(c.get('doc_id',''))} | page={c.get('page_num')}]"
        parts.append(header)
        parts.append(c.get("content", "").strip())
    parts.append("")
    parts.append(f"QUESTION: {question}")
    prompt = "\n".join(parts)
    return prompt


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def _find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = True) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _to_pixel_values(image: Image.Image, input_size: int = 448, max_num: int = 12, device: str = "cpu") -> torch.Tensor:
    transform = _build_transform(input_size=input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(tile) for tile in tiles]
    tensor = torch.stack(pixel_values)
    return tensor


@dataclass
class InternVL:
    model_id: str = "OpenGVLab/InternVL3_5-2B"
    device: str = "cuda"  # "cuda" fortement recommandé pour ce modèle
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    _loaded: bool = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoProcessor

        # Auto-pick device if unavailable
        try:
            if self.device == "cuda" and not torch.cuda.is_available():
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
        except Exception:
            self.device = "cpu"

        # Pick dtype by device if unspecified/unsafe
        desired_dtype = (self.dtype or "").lower()
        if self.device == "cpu":
            torch_dtype = torch.float32
        elif self.device == "mps":
            torch_dtype = torch.float16 if desired_dtype in {"", "float16", "bfloat16"} else (torch.float16 if desired_dtype == "float16" else torch.float16)
        else:
            torch_dtype = torch.bfloat16 if desired_dtype in {"", "bfloat16"} else (torch.float16 if desired_dtype == "float16" else torch.bfloat16)

        use_flash = bool(torch.cuda.is_available() and self.device == "cuda")

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        # Processor (si disponible) pour calculer pixel_values
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        except Exception:
            self.processor = None
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash,
            trust_remote_code=self.trust_remote_code,
        ).eval().to(self.device)
        logger.info("[internvl] loaded %s on device=%s in %.1f s", self.model_id, self.device, (time.perf_counter() - t0))
        self._loaded = True

    def generate(
        self,
        question: str,
        contexts: List[dict],
        run_model: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict:
        """Prépare l'entrée multimodale et tente d'appeler le modèle si `run_model=True`.
        Sinon, retourne un dry-run contenant le prompt et la liste d'images.
        """
        # Images à fournir au modèle (pages associées)
        image_paths = [c.get("image_path") for c in contexts if c.get("image_path")]
        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                pass

        prompt = build_prompt(question, contexts)

        if not run_model:
            return {
                "dry_run": True,
                "prompt": prompt,
                "num_images": len(images),
                "image_paths": image_paths,
            }

        # Tentative d'appel modèle (API exacte dépend du trust_remote_code)
        self._ensure_loaded()
        try:
            # De nombreux modèles InternVL exposent une méthode `chat` via trust_remote_code
            if hasattr(self.model, "chat"):
                import inspect
                t1 = time.perf_counter()
                main_image = images[0] if images else None
                # Build a plain dict as generation_config per official examples
                gen_cfg = {
                    "max_new_tokens": int(max_new_tokens),
                    "do_sample": bool(temperature > 0),
                    "temperature": float(temperature),
                }
                sig = None
                try:
                    sig = inspect.signature(self.model.chat)
                except Exception:
                    sig = None

                text = None

                # Prépare pixel_values si processor disponible
                pixel_values = None
                if main_image is not None:
                    try:
                        pv = _to_pixel_values(main_image, input_size=448, max_num=12)
                        # Match dtype to device
                        if self.device == "cuda":
                            pv = pv.to(torch.bfloat16)
                        else:
                            pv = pv.to(torch.float16)
                        pixel_values = pv.to(self.device)
                    except Exception:
                        pixel_values = None

                # Compose question: prefix <image> if we pass image tiles
                question_text = ("<image>\n" + question) if pixel_values is not None else question

                # Always call with positional signature (tokenizer, pixel_values_or_none, question, generation_config)
                try:
                    out = self.model.chat(self.tokenizer, pixel_values if pixel_values is not None else None, question_text, gen_cfg)
                    text = out if isinstance(out, str) else str(out)
                except Exception as e:
                    text = f"[Erreur d'inférence: {e}]"

                used_images = 1 if main_image and isinstance(text, str) and not text.startswith("[Erreur") else 0
                logger.info("[internvl] chat done tokens<=%d images_used=%d in %.1f s", max_new_tokens, used_images, (time.perf_counter() - t1))
            else:
                # Fallback très générique: concatène description d'images et prompt
                # Note: ceci ne reflète pas le vrai passage d'images si le modèle n'expose pas de pipeline adapté.
                full_prompt = PROMPT_SYSTEM + "\n\n" + prompt
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                t2 = time.perf_counter()
                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature)
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                logger.info("[internvl] generate done tokens<=%d images=%d in %.1f s", max_new_tokens, len(images), (time.perf_counter() - t2))
        except Exception as e:
            text = f"[Erreur d'inférence: {e}]"

        return {
            "dry_run": False,
            "prompt": prompt,
            "num_images": len(images),
            "image_paths": image_paths,
            "answer": text,
        }

