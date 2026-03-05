"""LLM augmentation module for Dance pipeline.

Provides audio understanding via Qwen2-Audio for:
- Rich tagging (subgenre, mood, notable elements)
- Cue point context generation
- Quality validation (BPM/key verification)
"""

from dance.llm.qwen_audio import QwenAudioModel, LLMAnalysisResult

__all__ = ["QwenAudioModel", "LLMAnalysisResult"]
