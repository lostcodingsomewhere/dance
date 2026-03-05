"""Tests for LLM augmentation module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dance.llm.qwen_audio import LLMAnalysisResult, QwenAudioModel
from dance.llm.prompts import COMBINED_ANALYSIS_PROMPT


class TestLLMAnalysisResult:
    """Tests for LLMAnalysisResult dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        result = LLMAnalysisResult()

        assert result.subgenre is None
        assert result.mood_tags == []
        assert result.notable_elements == []
        assert result.cue_contexts == {}
        assert result.bpm_validated is None
        assert result.error is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LLMAnalysisResult(
            subgenre="tech house",
            mood_tags=["dark", "driving"],
            notable_elements=["acid line"],
            cue_contexts={"intro": "Minimal kick"},
            bpm_validated=True,
            model_name="test-model",
        )

        d = result.to_dict()

        assert d["subgenre"] == "tech house"
        assert d["mood_tags"] == ["dark", "driving"]
        assert d["notable_elements"] == ["acid line"]
        assert d["cue_contexts"] == {"intro": "Minimal kick"}
        assert d["bpm_validated"] is True
        assert d["model_name"] == "test-model"

    def test_to_dict_with_none_values(self):
        """Test that None values are preserved in dict."""
        result = LLMAnalysisResult()
        d = result.to_dict()

        assert d["subgenre"] is None
        assert d["bpm_suggestion"] is None
        assert d["key_suggestion"] is None


class TestQwenAudioModel:
    """Tests for QwenAudioModel wrapper."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        model = QwenAudioModel()

        assert model.model_name == "Qwen/Qwen2-Audio-7B-Instruct"
        assert model.device == "auto"
        assert model.quantization == "4bit"
        assert model._loaded is False

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        model = QwenAudioModel(
            model_name="custom/model",
            device="mps",
            quantization="8bit",
            cache_dir=Path("/tmp/cache"),
        )

        assert model.model_name == "custom/model"
        assert model.device == "mps"
        assert model.quantization == "8bit"
        assert model.cache_dir == Path("/tmp/cache")

    @patch("dance.llm.qwen_audio.torch")
    def test_detect_device_auto_cuda(self, mock_torch):
        """Test device auto-detection prefers CUDA."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = True

        model = QwenAudioModel(device="auto")
        device = model._detect_device()

        assert device == "cuda"

    @patch("dance.llm.qwen_audio.torch")
    def test_detect_device_auto_mps(self, mock_torch):
        """Test device auto-detection falls back to MPS."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        model = QwenAudioModel(device="auto")
        device = model._detect_device()

        assert device == "mps"

    @patch("dance.llm.qwen_audio.torch")
    def test_detect_device_auto_cpu(self, mock_torch):
        """Test device auto-detection falls back to CPU."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        model = QwenAudioModel(device="auto")
        device = model._detect_device()

        assert device == "cpu"

    def test_detect_device_explicit(self):
        """Test explicit device setting is respected."""
        model = QwenAudioModel(device="cpu")
        device = model._detect_device()

        assert device == "cpu"

    def test_unload(self):
        """Test model unloading."""
        model = QwenAudioModel()
        model._model = MagicMock()
        model._processor = MagicMock()
        model._loaded = True

        with patch("dance.llm.qwen_audio.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            model.unload()

        assert model._model is None
        assert model._processor is None
        assert model._loaded is False

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        model = QwenAudioModel()
        result = LLMAnalysisResult()

        response = '''
        Here is the analysis:
        {
            "tagging": {
                "subgenre": "melodic techno",
                "mood_tags": ["dark", "hypnotic"],
                "notable_elements": ["arpeggios", "pads"],
                "energy_curve": "builds gradually",
                "dj_notes": "peak-time track"
            },
            "cue_contexts": {
                "intro": "Ambient pads",
                "drop_1": "Main melody"
            },
            "validation": {
                "bpm_correct": true,
                "bpm_suggestion": null,
                "key_correct": false,
                "key_suggestion": "Am",
                "quality_issues": [],
                "is_dj_track": true
            }
        }
        '''

        result = model._parse_response(response, result)

        assert result.subgenre == "melodic techno"
        assert result.mood_tags == ["dark", "hypnotic"]
        assert result.notable_elements == ["arpeggios", "pads"]
        assert result.cue_contexts["intro"] == "Ambient pads"
        assert result.bpm_validated is True
        assert result.key_validated is False
        assert result.key_suggestion == "Am"
        assert result.is_dj_track is True

    def test_parse_response_no_json(self):
        """Test handling response without JSON."""
        model = QwenAudioModel()
        result = LLMAnalysisResult()

        response = "This is not a JSON response at all."
        result = model._parse_response(response, result)

        assert result.error == "No JSON found in response"

    def test_parse_response_invalid_json(self):
        """Test handling invalid JSON."""
        model = QwenAudioModel()
        result = LLMAnalysisResult()

        response = '{"broken": json'
        result = model._parse_response(response, result)

        assert "JSON parse error" in result.error

    def test_parse_response_partial_data(self):
        """Test parsing response with partial data."""
        model = QwenAudioModel()
        result = LLMAnalysisResult()

        response = '''
        {
            "tagging": {
                "subgenre": "tech house"
            }
        }
        '''

        result = model._parse_response(response, result)

        assert result.subgenre == "tech house"
        assert result.mood_tags == []  # Default empty list
        assert result.cue_contexts == {}  # Default empty dict
        assert result.bpm_validated is None  # Not set

    @patch("dance.llm.qwen_audio.torch")
    def test_get_status(self, mock_torch):
        """Test getting model status."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        model = QwenAudioModel(device="mps", quantization="4bit")
        status = model.get_status()

        assert status["model_name"] == "Qwen/Qwen2-Audio-7B-Instruct"
        assert status["device"] == "mps"
        assert status["quantization"] == "4bit"
        assert status["loaded"] is False
        assert status["mps_available"] is True


class TestPrompts:
    """Tests for prompt templates."""

    def test_combined_prompt_formatting(self):
        """Test that combined prompt formats correctly."""
        prompt = COMBINED_ANALYSIS_PROMPT.format(
            bpm=128.5,
            key="Am",
            camelot_key="8A",
            energy=7,
        )

        assert "128.5" in prompt
        assert "Am" in prompt
        assert "8A" in prompt
        assert "7" in prompt
        assert "subgenre" in prompt
        assert "cue_contexts" in prompt
        assert "validation" in prompt

    def test_combined_prompt_unknown_values(self):
        """Test prompt with unknown values."""
        prompt = COMBINED_ANALYSIS_PROMPT.format(
            bpm="unknown",
            key="unknown",
            camelot_key="unknown",
            energy="unknown",
        )

        assert "unknown" in prompt


class TestDatabaseIntegration:
    """Tests for database schema integration."""

    def test_analysis_llm_fields_exist(self):
        """Test that Analysis model has LLM fields."""
        from dance.core.database import Analysis

        # Check that the columns exist (they're defined in the model)
        assert hasattr(Analysis, "llm_subgenre")
        assert hasattr(Analysis, "llm_mood_tags")
        assert hasattr(Analysis, "llm_notable_elements")
        assert hasattr(Analysis, "llm_energy_curve")
        assert hasattr(Analysis, "llm_dj_notes")
        assert hasattr(Analysis, "llm_cue_contexts")
        assert hasattr(Analysis, "llm_bpm_validated")
        assert hasattr(Analysis, "llm_key_validated")
        assert hasattr(Analysis, "llm_quality_issues")
        assert hasattr(Analysis, "llm_is_dj_track")
        assert hasattr(Analysis, "llm_model")
        assert hasattr(Analysis, "llm_analyzed_at")

    def test_track_state_llm_states_exist(self):
        """Test that TrackState has LLM states."""
        from dance.core.database import TrackState

        assert hasattr(TrackState, "LLM_AUGMENTING")
        assert hasattr(TrackState, "LLM_AUGMENTED")
        assert TrackState.LLM_AUGMENTING.value == "llm_augmenting"
        assert TrackState.LLM_AUGMENTED.value == "llm_augmented"


class TestTraktorExportIntegration:
    """Tests for Traktor export with LLM data."""

    def test_build_comment_with_llm_data(self):
        """Test comment building with LLM data."""
        from dance.export.traktor import TraktorExporter
        from dance.core.database import Analysis

        # Create mock analysis with LLM data
        analysis = MagicMock(spec=Analysis)
        analysis.llm_subgenre = "tech house"
        analysis.llm_mood_tags = json.dumps(["dark", "driving", "groovy"])
        analysis.llm_dj_notes = "Great peak-time track"
        analysis.floor_energy = 8
        analysis.bpm = 128.0
        analysis.key_camelot = "8A"
        analysis.mood_dark = None
        analysis.mood_aggressive = None

        exporter = TraktorExporter(MagicMock())
        comment = exporter._build_comment(analysis)

        assert "[Tech House]" in comment
        assert "dark" in comment
        assert "driving" in comment
        assert "Great peak-time track" in comment
        assert "E8" in comment
        assert "128bpm" in comment
        assert "8A" in comment

    def test_build_comment_without_llm_data(self):
        """Test comment building falls back when no LLM data."""
        from dance.export.traktor import TraktorExporter
        from dance.core.database import Analysis

        # Create mock analysis without LLM data
        analysis = MagicMock(spec=Analysis)
        analysis.llm_subgenre = None
        analysis.llm_mood_tags = None
        analysis.llm_dj_notes = None
        analysis.floor_energy = 7
        analysis.bpm = 126.0
        analysis.key_camelot = "11A"
        analysis.mood_dark = 0.7
        analysis.mood_aggressive = 0.8

        exporter = TraktorExporter(MagicMock())
        comment = exporter._build_comment(analysis)

        assert "dark" in comment  # From mood_dark
        assert "driving" in comment  # From mood_aggressive
        assert "E7" in comment
        assert "126bpm" in comment
