"""FoundationStereo / Fast-FoundationStereo inference backend."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FoundationStereoBackend:
    """
    Wrapper for NVlabs/FoundationStereo or NVlabs/Fast-FoundationStereo.

    All repo-specific imports stay in this module.
    """

    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        variant: str = "fast",
        device: str = "cuda",
        mixed_precision: bool = True,
        max_input_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.variant = variant.lower()
        self.device = device
        self.mixed_precision = mixed_precision
        self.max_input_size = max_input_size
        self._model = None

        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"FoundationStereo repo not found: {self.repo_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        self._load_model()

    def _load_model(self) -> None:
        """Try common FoundationStereo / Fast-FoundationStereo entry points."""
        if os.environ.get("FOUNDATIONSTEREO_MOCK") == "1":
            self._model = "mock"
            return

        errors: list[str] = []
        for mod_name, cls_name in (
            ("core.foundation_stereo", "FoundationStereo"),
            ("foundation_stereo", "FoundationStereo"),
            ("fs_model", "FoundationStereo"),
            ("core.model", "FoundationStereo"),
        ):
            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                self._model = cls(
                    self.checkpoint_path,
                    device=self.device,
                    mixed_precision=self.mixed_precision,
                    variant=self.variant,
                )
                return
            except Exception as exc:
                errors.append(f"{mod_name}.{cls_name}: {exc}")

        # Fallback: script-style inference module
        infer_path = self.repo_path / "inference.py"
        if infer_path.is_file():
            spec = importlib.util.spec_from_file_location("fs_inference", infer_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "load_model"):
                    self._model = mod.load_model(
                        str(self.checkpoint_path), device=self.device, variant=self.variant
                    )
                    return
                if hasattr(mod, "FoundationStereo"):
                    self._model = mod.FoundationStereo(str(self.checkpoint_path), device=self.device)
                    return

        raise ImportError(
            "Could not load FoundationStereo from "
            f"{self.repo_path}. Tried imports: {errors}. "
            "Set FOUNDATIONSTEREO_MOCK=1 for tests or install the repo per NVlabs instructions."
        )

    def _resize_pair(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if self.max_input_size is None:
            return left, right, 1.0
        max_w, max_h = self.max_input_size
        h, w = left.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale >= 1.0:
            return left, right, 1.0
        nw, nh = int(w * scale), int(h * scale)
        left_r = cv2.resize(left, (nw, nh), interpolation=cv2.INTER_LINEAR)
        right_r = cv2.resize(right, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return left_r, right_r, scale

    def predict_disparity(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Disparity in pixels at original left-image width."""
        if left_rgb.dtype != np.uint8:
            left_rgb = np.clip(left_rgb, 0, 255).astype(np.uint8)
        if right_rgb.dtype != np.uint8:
            right_rgb = np.clip(right_rgb, 0, 255).astype(np.uint8)

        orig_h, orig_w = left_rgb.shape[:2]
        left_in, right_in, scale = self._resize_pair(left_rgb, right_rgb)

        if self._model == "mock":
            # Gradient disparity for tests: closer objects -> larger disparity
            gray = cv2.cvtColor(left_in, cv2.COLOR_RGB2GRAY).astype(np.float32)
            disp = np.maximum(1.0, 80.0 - gray * 0.2)
            disp[gray < 5] = 0
        elif hasattr(self._model, "infer"):
            disp = self._model.infer(left_in, right_in)
        elif hasattr(self._model, "predict"):
            disp = self._model.predict(left_in, right_in)
        elif callable(self._model):
            disp = self._model(left_in, right_in)
        else:
            raise RuntimeError("Loaded model has no infer/predict/call interface")

        disp = np.asarray(disp, dtype=np.float32)
        if disp.ndim == 3:
            disp = disp.squeeze()
        if scale != 1.0:
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / scale
        return disp.astype(np.float32)

    def predict_batch(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        return [self.predict_disparity(l, r) for l, r in pairs]
