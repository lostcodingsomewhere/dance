"""Prompt templates for Qwen2-Audio DJ track analysis."""

TAGGING_PROMPT = """Analyze this electronic dance music track and provide the following information in JSON format:

{{
    "subgenre": "specific subgenre (e.g., tech house, melodic techno, progressive house, minimal, deep house, trance, drum and bass)",
    "mood_tags": ["list", "of", "mood", "descriptors"],
    "notable_elements": ["list", "of", "notable", "musical", "elements"],
    "energy_curve": "description of how energy changes throughout the track",
    "dj_notes": "brief notes for a DJ about when/how to use this track"
}}

For mood_tags, choose from: dark, uplifting, driving, hypnotic, groovy, aggressive, euphoric, melancholic, minimal, atmospheric, industrial, warm, cold, trippy, percussive, melodic

For notable_elements, identify things like: vocal chops, acid line, big room synths, rolling bassline, arpeggios, pads, strings, tribal percussion, glitchy sounds, filter sweeps, risers, sub bass, stabs, etc.

For dj_notes, consider: Is this a good opener, peak-time track, or closer? Does it have a long intro for mixing? Any tricky transitions to watch for?

Current analysis data:
- BPM: {bpm}
- Key: {key} (Camelot: {camelot_key})
- Energy: {energy}/10

Respond ONLY with valid JSON, no additional text."""


CUE_CONTEXT_PROMPT = """Analyze this electronic dance music track and identify the main sections. For each section you can identify, provide a brief musical description.

Respond in JSON format:
{{
    "intro": "what's happening in the intro (e.g., 'Minimal kick pattern with hi-hats')",
    "buildup_1": "description of first buildup if present",
    "drop_1": "description of first drop/main section",
    "breakdown": "description of breakdown section if present",
    "buildup_2": "description of second buildup if present",
    "drop_2": "description of second drop if present",
    "outro": "what's happening in the outro"
}}

Keep descriptions SHORT (under 30 characters) - they will be used as cue point names in DJ software.
Only include sections that are clearly present in the track.
Focus on the most notable sonic characteristic of each section.

Current analysis data:
- BPM: {bpm}
- Key: {key}
- Energy: {energy}/10

Respond ONLY with valid JSON, no additional text."""


VALIDATION_PROMPT = """Listen to this electronic dance music track and validate the following analysis:

- Detected BPM: {bpm}
- Detected Key: {key} (Camelot: {camelot_key})

Respond in JSON format:
{{
    "bpm_correct": true or false,
    "bpm_suggestion": null or corrected BPM if incorrect,
    "key_correct": true or false,
    "key_suggestion": null or corrected key if incorrect,
    "quality_issues": ["list", "of", "any", "audio", "quality", "issues"],
    "is_dj_track": true or false,
    "notes": "any additional observations"
}}

For quality_issues, check for: clipping, distortion, poor encoding artifacts, mono issues, very low/high volume, etc.

For is_dj_track, determine if this is actually a DJ-mixable track (has consistent beat, proper structure) vs something like an ambient piece, spoken word, jingle, etc.

Respond ONLY with valid JSON, no additional text."""


COMBINED_ANALYSIS_PROMPT = """Analyze this electronic dance music track comprehensively.

Current analysis data:
- BPM: {bpm}
- Key: {key} (Camelot: {camelot_key})
- Energy: {energy}/10

Provide your analysis in JSON format with these sections:

{{
    "tagging": {{
        "subgenre": "specific subgenre",
        "mood_tags": ["mood1", "mood2", "mood3"],
        "notable_elements": ["element1", "element2"],
        "energy_curve": "how energy changes through the track",
        "dj_notes": "when/how to use this track"
    }},
    "cue_contexts": {{
        "intro": "short description for cue name (<25 chars)",
        "drop_1": "short description",
        "breakdown": "short description if present",
        "drop_2": "short description if present",
        "outro": "short description"
    }},
    "validation": {{
        "bpm_correct": true/false,
        "bpm_suggestion": null or number,
        "key_correct": true/false,
        "key_suggestion": null or "key string",
        "quality_issues": [],
        "is_dj_track": true/false
    }}
}}

Guidelines:
- subgenre: tech house, melodic techno, progressive house, minimal, deep house, trance, etc.
- mood_tags: dark, uplifting, driving, hypnotic, groovy, aggressive, euphoric, melancholic, minimal, atmospheric
- cue_contexts: Keep SHORT (<25 chars) - used as DJ cue point names
- Only include cue sections that are clearly present

Respond ONLY with valid JSON."""
