// TypeScript mirrors of the Pydantic schemas in src/dance/api/schemas.py.
// Keep in sync if the contract changes.

export interface Analysis {
  bpm: number | null;
  key_camelot: string | null;
  key_standard: string | null;
  floor_energy: number | null;
  energy_overall: number | null;
  brightness: number | null;
  warmth: number | null;
  danceability: number | null;
}

export interface StemAnalysis {
  bpm: number | null;
  energy_overall: number | null;
  floor_energy: number | null;
  presence_ratio: number | null;
  vocal_present: boolean | null;
  kick_density: number | null;
  dominant_pitch_camelot: string | null;
}

export interface StemFile {
  id: number;
  kind: string;
  path: string;
  analysis: StemAnalysis | null;
}

export interface Region {
  id: number;
  position_ms: number;
  length_ms: number | null;
  region_type: string;
  section_label: string | null;
  length_bars: number | null;
  name: string | null;
  color: string | null;
  confidence: number | null;
  source: string;
  stem_file_id: number | null;
}

export interface Track {
  id: number;
  file_path: string;
  title: string | null;
  artist: string | null;
  duration_seconds: number | null;
  state: string;
  analysis: Analysis | null;
  tags: string[];
}

export interface Recommendation {
  track_id: number;
  score: number;
  reasons: { kind?: string; weight?: number; [k: string]: unknown }[];
  title: string | null;
  artist: string | null;
  bpm: number | null;
  key_camelot: string | null;
  floor_energy: number | null;
}

export interface RecommendRequest {
  seeds: number[];
  k?: number;
  kinds?: string[];
  weights?: Record<string, number>;
  exclude?: number[];
}

export interface SessionPlay {
  track_id: number;
  played_at: string; // ISO datetime
  position_in_set: number;
  energy_at_play: number | null;
  transition_type: string | null;
  title: string | null;
  artist: string | null;
}

export interface DjSession {
  id: number;
  name: string | null;
  notes: string | null;
  started_at: string;
  ended_at: string | null;
  plays: SessionPlay[];
}

export interface AddPlayBody {
  track_id: number;
  transition_type?: string | null;
  duration_played_ms?: number | null;
}

export interface AbletonState {
  tempo: number | null;
  is_playing: boolean | null;
  beat: number | null;
  playing_clips: Record<string, number>;
  track_volumes: Record<string, number>;
}

export const EMPTY_ABLETON_STATE: AbletonState = {
  tempo: null,
  is_playing: null,
  beat: null,
  playing_clips: {},
  track_volumes: {},
};

export interface TrackFilters {
  limit?: number;
  offset?: number;
  bpm_min?: number;
  bpm_max?: number;
  key?: string;
  energy?: number;
  state?: string;
}

export type ViewName = "now" | "next" | "library" | "session";
