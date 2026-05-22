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
  file_path: string | null;
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

// Per-column rec stream — five parallel streams that re-score as the
// active combo changes. ``stem_file_id`` is null for the mix column.
export interface ColumnRec {
  track_id: number;
  stem_file_id: number | null;
  track_title: string | null;
  track_artist: string | null;
  bpm: number | null;
  key_camelot: string | null;
  floor_energy: number | null;
  score: number;
  score_breakdown: Record<string, number>;
  reasons: string[];
}

export interface ColumnRecsResponse {
  column: string;
  combo_size: number;
  recs: ColumnRec[];
}

export interface ColumnRecsRequest {
  column: string;
  combo_stem_ids: number[];
  master_bpm?: number | null;
  k?: number;
  exclude_track_ids?: number[];
}

export interface PreviewRequest {
  track_id: number;
  column: string;
}

export interface PreviewResult {
  ok: boolean;
  cue_track_idx?: number | null;
  slot?: number | null;
  audio_path?: string | null;
  label?: string | null;
  warnings?: string[];
}

// Mirrors src/dance/api/schemas.py::WaveformOut. Pre-decimated amplitude
// envelope returned by GET /tracks/{id}/waveform and /stems/{id}/waveform.
export interface Waveform {
  peaks: number[];
  duration_seconds: number;
  num_peaks: number;
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
  /** track_index → output meter level 0-1. Subscribed on deck columns. */
  track_meters: Record<string, number>;
  /** track_index → currently-playing clip's playing_position in beats. */
  playing_positions: Record<string, number>;
}

export const EMPTY_ABLETON_STATE: AbletonState = {
  tempo: null,
  is_playing: null,
  beat: null,
  playing_clips: {},
  track_volumes: {},
  track_meters: {},
  playing_positions: {},
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

export type ViewName = "booth" | "crate" | "pipeline";

/**
 * A track loaded into Live via "Load to Live". Records which Ableton scene the
 * stems were placed into so we can detect "this track is playing now" by
 * cross-referencing AbletonState.playing_clips (which is keyed by
 * ableton_track_idx → scene_idx).
 *
 * stem_track_indices is the set of ableton track indices that host this dance
 * track's stems. If any of those tracks shows a playing clip whose scene_idx
 * equals scene_index, this dance track is the one playing.
 */
export interface LoadedDeck {
  track_id: number;
  scene_index: number;
  stem_track_indices: number[];
  loaded_at: number; // ms epoch
}

// Mirrors src/dance/api/schemas.py::DeckCellOut + DeckMapOut. The bridge
// is the source of truth for which cells (scene × stem-kind) are loaded
// from which source tracks. The FE polls /ableton/decks to render the
// SceneGrid + ComboStrip.
export interface DeckCell {
  scene_index: number;
  kind: string; // "drums" | "bass" | "vocals" | "other"
  track_id: number;
  stem_file_id: number | null;
  title: string | null;
  artist: string | null;
  bpm: number | null;
  key_camelot: string | null;
  floor_energy: number | null;
}

export interface DeckMap {
  columns: Record<string, number> | null;
  cells: DeckCell[];
}

// Pipeline-ops mirror of `src/dance/api/schemas.py:PipelineStatusOut`.
export interface PipelineStatus {
  counts: Record<string, number>; // keyed by TrackState enum value
  total: number;
  in_progress: boolean;
  errors: number;
  complete: number;
  weighted_progress: number; // 0-100, stage-weighted
}

// Pipeline-ops mirror of `src/dance/api/schemas.py:PipelineRecentTrackOut`.
export interface PipelineRecentTrack {
  id: number;
  title: string | null;
  artist: string | null;
  state: string;
  updated_at: string | null; // ISO datetime
  error_message: string | null;
}

// Ordered stages — the UI renders the count grid left-to-right in this order
// so the eye can read pipeline progress like a Gantt chart.
//
// LEGACY: kept for backward-compat references in other components. The Ops
// view uses the consolidated STAGE_GROUPS below.
export const PIPELINE_STAGES: { key: string; label: string }[] = [
  { key: "pending", label: "Pending" },
  { key: "analyzing", label: "Analyzing" },
  { key: "analyzed", label: "Analyzed" },
  { key: "separating", label: "Separating" },
  { key: "separated", label: "Separated" },
  { key: "analyzing_stems", label: "Stem analyze" },
  { key: "stems_analyzed", label: "Stems done" },
  { key: "detecting_regions", label: "Regions…" },
  { key: "regions_detected", label: "Regions ✓" },
  { key: "embedding", label: "Embedding" },
  { key: "embedded", label: "Embedded" },
  { key: "complete", label: "Complete" },
  { key: "error", label: "Error" },
];

/**
 * Pipeline stages, consolidated. The DB has 13 ``TrackState`` values that
 * encode (waiting-for-stage, in-flight-in-stage) pairs; the UI wants to think
 * about 5 stages + done + error, each with two sub-counts:
 *
 * - ``waiting_state`` — tracks queued for the stage but not yet picked up.
 * - ``active_state`` — tracks a worker is currently processing.
 *
 * The "done" bucket folds ``embedded`` and ``complete`` together: in the
 * current pipeline both mean "finished embed, no work left" — the
 * distinction is just that ``embedded`` is the embed stage's output_state
 * and ``complete`` is what advances later if there's a follow-up step.
 */
export interface StageGroup {
  key: string;
  label: string;
  waiting_state: string | null; // null for done/error
  active_state: string | null;
  color: string; // tailwind classes for the tile
}

export const STAGE_GROUPS: StageGroup[] = [
  {
    key: "analyze",
    label: "Analyze",
    waiting_state: "pending",
    active_state: "analyzing",
    color: "bg-amber-500/10 text-amber-200 ring-1 ring-amber-500/20",
  },
  {
    key: "separate",
    label: "Separate",
    waiting_state: "analyzed",
    active_state: "separating",
    color: "bg-orange-500/10 text-orange-200 ring-1 ring-orange-500/20",
  },
  {
    key: "analyze_stems",
    label: "Stems",
    waiting_state: "separated",
    active_state: "analyzing_stems",
    color: "bg-blue-500/10 text-blue-200 ring-1 ring-blue-500/20",
  },
  {
    key: "detect_regions",
    label: "Regions",
    waiting_state: "stems_analyzed",
    active_state: "detecting_regions",
    color: "bg-purple-500/10 text-purple-200 ring-1 ring-purple-500/20",
  },
  {
    key: "embed",
    label: "Embed",
    waiting_state: "regions_detected",
    active_state: "embedding",
    color: "bg-cyan-500/10 text-cyan-200 ring-1 ring-cyan-500/20",
  },
];

/** Terminal buckets, rendered at the end of the row.
 *
 * ``states`` is the set of TrackState values whose counts roll up into this
 * tile. ``filter_state`` is the single state used when the tile is clicked —
 * the activity table can only filter on one state at a time, and we want
 * clicking to surface the bucket the user expects. (The ``embed`` stage's
 * output_state is COMPLETE directly — EMBEDDED is unused in the current
 * pipeline, so a "Done" click that filters on EMBEDDED would show an empty
 * table.) */
export const TERMINAL_GROUPS: {
  key: string;
  label: string;
  states: string[];
  filter_state: string;
  color: string;
}[] = [
  {
    key: "done",
    label: "Done",
    states: ["embedded", "complete"], // sum counts (embedded is 0 in practice)
    filter_state: "complete", // click → show the actually-populated state
    color: "bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-500/30",
  },
  {
    key: "error",
    label: "Error",
    states: ["error"],
    filter_state: "error",
    color: "bg-rose-500/20 text-rose-200 ring-1 ring-rose-500/30",
  },
];

/** Computed view of a stage group given the raw counts dict. */
export interface StageGroupCounts {
  waiting: number;
  active: number;
  total: number;
}

export function computeStageCounts(
  group: StageGroup,
  counts: Record<string, number>,
): StageGroupCounts {
  const waiting = group.waiting_state ? counts[group.waiting_state] ?? 0 : 0;
  const active = group.active_state ? counts[group.active_state] ?? 0 : 0;
  return { waiting, active, total: waiting + active };
}

/** Map a raw TrackState value to a user-readable filter description, so the
 * activity panel header reads "Waiting for Separate" instead of
 * "Tracks in 'analyzed'". The state-key naming is internal pipeline
 * vocabulary; users think in stage names. */
export function describeFilterState(stateKey: string): string {
  for (const g of STAGE_GROUPS) {
    if (g.waiting_state === stateKey) return `Waiting for ${g.label}`;
    if (g.active_state === stateKey) return `In ${g.label}`;
  }
  if (stateKey === "complete") return "Complete";
  if (stateKey === "embedded") return "Embedded";
  if (stateKey === "error") return "Errored";
  return stateKey;
}

export type OpsViewMode = "grid" | "board";

// CSV ingest types — mirror `src/dance/api/schemas.py` Ingest* classes.

export interface IngestPreviewRow {
  artist: string;
  title: string;
  album: string;
  duration_s: number;
  target_path: string;
  target_exists: boolean;
  duplicate_of: number | null;
}

export interface IngestPreviewResponse {
  total_rows: number;
  new_rows: IngestPreviewRow[];
  duplicates: IngestPreviewRow[];
  parse_errors: string[];
}

export interface JobItem {
  label: string;
  status: "pending" | "ok" | "skip" | "fail";
  message: string;
}

export interface Job {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  revision: number;
  total: number;
  counts: { pending: number; ok: number; skip: number; fail: number };
  items: JobItem[];
}
