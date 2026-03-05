"""Qwen2-Audio model wrapper for DJ track analysis.

Provides native audio understanding without transcription for:
- Rich tagging (subgenre, mood, notable elements)
- Cue point context generation
- Quality validation (BPM/key verification)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    """Result of LLM audio analysis."""

    # Tagging
    subgenre: str | None = None
    mood_tags: list[str] = field(default_factory=list)
    notable_elements: list[str] = field(default_factory=list)
    energy_curve: str | None = None
    dj_notes: str | None = None

    # Cue contexts (keyed by cue type: intro, drop_1, breakdown, etc.)
    cue_contexts: dict[str, str] = field(default_factory=dict)

    # Validation
    bpm_validated: bool | None = None
    bpm_suggestion: float | None = None
    key_validated: bool | None = None
    key_suggestion: str | None = None
    quality_issues: list[str] = field(default_factory=list)
    is_dj_track: bool | None = None

    # Metadata
    model_name: str | None = None
    raw_response: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "subgenre": self.subgenre,
            "mood_tags": self.mood_tags,
            "notable_elements": self.notable_elements,
            "energy_curve": self.energy_curve,
            "dj_notes": self.dj_notes,
            "cue_contexts": self.cue_contexts,
            "bpm_validated": self.bpm_validated,
            "bpm_suggestion": self.bpm_suggestion,
            "key_validated": self.key_validated,
            "key_suggestion": self.key_suggestion,
            "quality_issues": self.quality_issues,
            "is_dj_track": self.is_dj_track,
            "model_name": self.model_name,
        }


class QwenAudioModel:
    """Wrapper for Qwen2-Audio model with lazy loading and MPS support."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "auto",
        quantization: str | None = "4bit",
        cache_dir: Path | None = None,
    ):
        """Initialize the model wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("auto", "mps", "cuda", "cpu")
            quantization: Quantization level ("4bit", "8bit", or None for full precision)
            cache_dir: Optional cache directory for model weights
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.cache_dir = cache_dir

        self._model = None
        self._processor = None
        self._loaded = False

    def _detect_device(self) -> str:
        """Detect the best available device."""
        import torch

        if self.device != "auto":
            return self.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self) -> None:
        """Lazily load the model and processor."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "LLM dependencies not installed. Install with: pip install dance[llm]"
            ) from e

        device = self._detect_device()
        logger.info(f"Loading {self.model_name} on {device} (quantization: {self.quantization})")

        # Configure quantization
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }

        if self.cache_dir:
            model_kwargs["cache_dir"] = str(self.cache_dir)

        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to full precision")
                model_kwargs["torch_dtype"] = torch.float16
        elif self.quantization == "8bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to full precision")
                model_kwargs["torch_dtype"] = torch.float16
        else:
            # Full precision (or float16 for memory efficiency)
            model_kwargs["torch_dtype"] = torch.float16

        # Set device map
        if device == "mps":
            # MPS doesn't support device_map="auto", load to CPU then move
            model_kwargs["device_map"] = None
        else:
            model_kwargs["device_map"] = "auto"

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Move to MPS if needed
        if device == "mps" and "quantization_config" not in model_kwargs:
            self._model = self._model.to("mps")

        self._model.eval()
        self._loaded = True
        logger.info(f"Model loaded successfully on {device}")

    def is_available(self) -> bool:
        """Check if the model can be loaded."""
        try:
            import torch
            from transformers import AutoProcessor
            return True
        except ImportError:
            return False

    def analyze(
        self,
        audio_path: Path,
        bpm: float | None = None,
        key: str | None = None,
        camelot_key: str | None = None,
        energy: float | None = None,
    ) -> LLMAnalysisResult:
        """Analyze an audio file and return rich metadata.

        Args:
            audio_path: Path to the audio file
            bpm: Detected BPM from Essentia analysis
            key: Detected key from Essentia analysis
            camelot_key: Camelot wheel notation of the key
            energy: Energy score (1-10) from Essentia analysis

        Returns:
            LLMAnalysisResult with tagging, cue contexts, and validation
        """
        from dance.llm.prompts import COMBINED_ANALYSIS_PROMPT

        self._load_model()

        result = LLMAnalysisResult(model_name=self.model_name)

        try:
            import librosa
            import torch

            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

            # Format the prompt with analysis data
            prompt = COMBINED_ANALYSIS_PROMPT.format(
                bpm=bpm or "unknown",
                key=key or "unknown",
                camelot_key=camelot_key or "unknown",
                energy=energy or "unknown",
            )

            # Prepare inputs for Qwen2-Audio
            # The model expects audio in a specific format
            inputs = self._processor(
                text=prompt,
                audios=[audio],
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            device = self._detect_device()
            if device == "mps":
                inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            elif device == "cuda":
                inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode response
            response = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
            result.raw_response = response

            # Parse JSON from response
            result = self._parse_response(response, result)

        except Exception as e:
            logger.error(f"Error analyzing {audio_path}: {e}")
            result.error = str(e)

        return result

    def _parse_response(self, response: str, result: LLMAnalysisResult) -> LLMAnalysisResult:
        """Parse the JSON response from the model."""
        try:
            # Find JSON in response (model might include extra text)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in model response")
                result.error = "No JSON found in response"
                return result

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Parse tagging section
            if "tagging" in data:
                tagging = data["tagging"]
                result.subgenre = tagging.get("subgenre")
                result.mood_tags = tagging.get("mood_tags", [])
                result.notable_elements = tagging.get("notable_elements", [])
                result.energy_curve = tagging.get("energy_curve")
                result.dj_notes = tagging.get("dj_notes")

            # Parse cue contexts
            if "cue_contexts" in data:
                result.cue_contexts = data["cue_contexts"]

            # Parse validation section
            if "validation" in data:
                validation = data["validation"]
                result.bpm_validated = validation.get("bpm_correct")
                result.bpm_suggestion = validation.get("bpm_suggestion")
                result.key_validated = validation.get("key_correct")
                result.key_suggestion = validation.get("key_suggestion")
                result.quality_issues = validation.get("quality_issues", [])
                result.is_dj_track = validation.get("is_dj_track")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            result.error = f"JSON parse error: {e}"

        return result

    def get_status(self) -> dict[str, Any]:
        """Get model status information."""
        import torch

        device = self._detect_device()

        status = {
            "model_name": self.model_name,
            "device": device,
            "quantization": self.quantization,
            "loaded": self._loaded,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }

        if device == "cuda" and torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"

        return status

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False

        # Clear CUDA/MPS cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded")
