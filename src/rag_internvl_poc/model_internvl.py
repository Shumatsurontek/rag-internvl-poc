from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image


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


@dataclass
class InternVL:
    model_id: str = "OpenGVLab/InternVL3_5-8B"
    device: str = "cuda"  # "cuda" fortement recommandé pour ce modèle
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    _loaded: bool = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel

        dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=self.trust_remote_code,
        ).eval().to(self.device)
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
        import torch

        try:
            # De nombreux modèles InternVL exposent une méthode `chat` via trust_remote_code
            if hasattr(self.model, "chat"):
                out = self.model.chat(
                    self.tokenizer,
                    prompt,
                    images=images if images else None,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                text = out if isinstance(out, str) else str(out)
            else:
                # Fallback très générique: concatène description d'images et prompt
                # Note: ceci ne reflète pas le vrai passage d'images si le modèle n'expose pas de pipeline adapté.
                full_prompt = PROMPT_SYSTEM + "\n\n" + prompt
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature)
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            text = f"[Erreur d'inférence: {e}]"

        return {
            "dry_run": False,
            "prompt": prompt,
            "num_images": len(images),
            "image_paths": image_paths,
            "answer": text,
        }

