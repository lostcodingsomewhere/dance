import type {
  AbletonState,
  AddPlayBody,
  ColumnRecsRequest,
  ColumnRecsResponse,
  DanceSet,
  DeckMap,
  DjSession,
  IngestPreviewResponse,
  IngestTrackResult,
  Job,
  PipelineRecentTrack,
  PipelineStatus,
  PreviewRequest,
  PreviewResult,
  Recommendation,
  RecommendRequest,
  Region,
  SetSummary,
  SpotifySearchResponse,
  SpotifyTrackHit,
  StemFile,
  TailRecsResponse,
  Track,
  TrackFilters,
  Waveform,
} from "./types";

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
    message?: string,
  ) {
    super(message ?? `API ${status}: ${body || "(empty body)"}`);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url = path.startsWith("http") ? path : `${BASE}${path}`;
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text);
  }
  // 204 No Content
  if (res.status === 204) return undefined as unknown as T;
  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("application/json")) {
    return undefined as unknown as T;
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, unknown>): string {
  const entries = Object.entries(params).filter(
    ([, v]) => v !== undefined && v !== null && v !== "",
  );
  if (entries.length === 0) return "";
  const search = new URLSearchParams();
  for (const [k, v] of entries) search.set(k, String(v));
  return `?${search.toString()}`;
}

// Tracks --------------------------------------------------------------------

export function getTracks(filters: TrackFilters = {}): Promise<Track[]> {
  return request<Track[]>(`/tracks${qs(filters as Record<string, unknown>)}`);
}

export function getTrack(id: number): Promise<Track> {
  return request<Track>(`/tracks/${id}`);
}

export interface SearchTracksFilters {
  q?: string;
  limit?: number;
  bpm_min?: number;
  bpm_max?: number;
  key?: string;
  energy?: number;
}

export function searchTracks(filters: SearchTracksFilters = {}): Promise<Track[]> {
  return request<Track[]>(
    `/tracks/search${qs(filters as Record<string, unknown>)}`,
  );
}

export function getRegions(trackId: number): Promise<Region[]> {
  return request<Region[]>(`/tracks/${trackId}/regions`);
}

export function getStems(trackId: number): Promise<StemFile[]> {
  return request<StemFile[]>(`/tracks/${trackId}/stems`);
}

export function getTrackWaveform(trackId: number, numPeaks = 200): Promise<Waveform> {
  return request<Waveform>(`/tracks/${trackId}/waveform?num_peaks=${numPeaks}`);
}

export function getStemWaveform(stemFileId: number, numPeaks = 200): Promise<Waveform> {
  return request<Waveform>(`/stems/${stemFileId}/waveform?num_peaks=${numPeaks}`);
}

// Recommend -----------------------------------------------------------------

export function recommend(req: RecommendRequest): Promise<Recommendation[]> {
  return request<Recommendation[]>("/recommend", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function recommendBySeed(
  id: number,
  k = 10,
): Promise<Recommendation[]> {
  return request<Recommendation[]>(`/recommend/by-seed/${id}?k=${k}`);
}

export function recommendByText(
  query: string,
  k = 20,
  exclude?: number[],
): Promise<Recommendation[]> {
  return request<Recommendation[]>("/recommend/text", {
    method: "POST",
    body: JSON.stringify({ query, k, exclude }),
  });
}

export function recommendByColumn(
  body: ColumnRecsRequest,
): Promise<ColumnRecsResponse> {
  return request<ColumnRecsResponse>("/recommend/by-column", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// Sessions ------------------------------------------------------------------

export function createSession(name?: string): Promise<DjSession> {
  return request<DjSession>("/sessions", {
    method: "POST",
    body: JSON.stringify({ name: name ?? null }),
  });
}

export async function currentSession(): Promise<DjSession | null> {
  try {
    return await request<DjSession>("/sessions/current");
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

export function getSession(id: number): Promise<DjSession> {
  return request<DjSession>(`/sessions/${id}`);
}

export function addPlay(
  sessionId: number,
  body: AddPlayBody,
): Promise<DjSession> {
  return request<DjSession>(`/sessions/${sessionId}/plays`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function endSession(sessionId: number): Promise<DjSession> {
  return request<DjSession>(`/sessions/${sessionId}/end`, { method: "POST" });
}

// Sets ----------------------------------------------------------------------

export function listSets(): Promise<SetSummary[]> {
  return request<SetSummary[]>("/sets");
}

export async function getActiveSet(): Promise<DanceSet | null> {
  try {
    return await request<DanceSet>("/sets/active");
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

export function getSet(id: number): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${id}`);
}

export function createSet(name: string, notes?: string): Promise<DanceSet> {
  return request<DanceSet>("/sets", {
    method: "POST",
    body: JSON.stringify({ name, notes: notes ?? null }),
  });
}

export function updateSet(
  id: number,
  patch: { name?: string; notes?: string | null },
): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${id}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function deleteSet(id: number): Promise<void> {
  return request<void>(`/sets/${id}`, { method: "DELETE" });
}

export function activateSet(id: number): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${id}/activate`, { method: "POST" });
}

export function addTrackToSet(
  setId: number,
  trackId: number,
  opts: { position?: number; note?: string; stem_kinds?: string[] | null } = {},
): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${setId}/tracks`, {
    method: "POST",
    body: JSON.stringify({
      track_id: trackId,
      position: opts.position ?? null,
      note: opts.note ?? null,
      stem_kinds: opts.stem_kinds ?? null,
    }),
  });
}

export function moveTrackInSet(
  setId: number,
  trackId: number,
  position: number,
): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${setId}/tracks/${trackId}`, {
    method: "PATCH",
    body: JSON.stringify({ position }),
  });
}

export function updateSetTrackNote(
  setId: number,
  trackId: number,
  note: string,
): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${setId}/tracks/${trackId}`, {
    method: "PATCH",
    body: JSON.stringify({ note }),
  });
}

/** Update a set track's stem filter. Pass null to clear (load all stems);
 *  pass a non-empty subset of ["drums","bass","vocals","other"] to limit. */
export function updateSetTrackStemKinds(
  setId: number,
  trackId: number,
  stemKinds: string[] | null,
): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${setId}/tracks/${trackId}`, {
    method: "PATCH",
    body: JSON.stringify({ stem_kinds: stemKinds }),
  });
}

export function removeTrackFromSet(
  setId: number,
  trackId: number,
): Promise<DanceSet> {
  return request<DanceSet>(`/sets/${setId}/tracks/${trackId}`, {
    method: "DELETE",
  });
}

export function getTailRecs(
  setId: number,
  opts: { k?: number; excludeSessionPlays?: boolean } = {},
): Promise<TailRecsResponse> {
  const params: Record<string, unknown> = {};
  if (opts.k != null) params.k = opts.k;
  if (opts.excludeSessionPlays) params.exclude_session_plays = "true";
  return request<TailRecsResponse>(`/sets/${setId}/tail-recs${qs(params)}`);
}

// Spotify search + optimistic ingest --------------------------------------

export function spotifySearch(
  q: string,
  limit = 8,
): Promise<SpotifySearchResponse> {
  return request<SpotifySearchResponse>(
    `/spotify/search${qs({ q, limit })}`,
  );
}

export function ingestSpotifyTrack(
  hit: SpotifyTrackHit,
): Promise<IngestTrackResult> {
  return request<IngestTrackResult>("/pipeline/ingest/track", {
    method: "POST",
    body: JSON.stringify({
      spotify_id: hit.spotify_id,
      title: hit.title,
      artist: hit.artist,
      duration_ms: hit.duration_ms,
    }),
  });
}

export interface IngestPlaylistResult {
  playlist_id: string;
  total_in_playlist: number;
  added: number;
  skipped_existing: number;
  /** Of ``skipped_existing``, how many had their spotify_id back-filled
   *  onto a previously-unlinked Track (because their (artist, title)
   *  fuzzy-matched a CSV-ingested track). */
  backfilled: number;
  failed: number;
  track_ids: number[];
  job_ids: string[];
}

/** Paste a Spotify playlist URL — backend fetches the track list via Web
 *  API, dedupes against the existing library, and spawns a yt-dlp
 *  download job for every new ID. See ``POST /pipeline/ingest/playlist``.
 *
 *  ``userToken``: optional Spotify USER OAuth token. Required in practice
 *  because Spotify's Nov 2024 API policy blocks Client Credentials apps
 *  from reading playlist tracks. Grab one from exportify.net DevTools
 *  (any /me/playlists request → Authorization header → copy Bearer
 *  value). Lives in memory only — never persisted to localStorage. */
export function ingestSpotifyPlaylist(
  playlistUrl: string,
  userToken?: string,
): Promise<IngestPlaylistResult> {
  return request<IngestPlaylistResult>("/pipeline/ingest/playlist", {
    method: "POST",
    body: JSON.stringify({
      playlist_url: playlistUrl,
      user_token: userToken || null,
    }),
  });
}

// Pipeline ops --------------------------------------------------------------

export function getPipelineStatus(): Promise<PipelineStatus> {
  return request<PipelineStatus>("/pipeline/status");
}

export function getPipelineRecent(
  limit = 20,
  state?: string | null,
): Promise<PipelineRecentTrack[]> {
  const q = state
    ? `?limit=${limit}&state=${encodeURIComponent(state)}`
    : `?limit=${limit}`;
  return request<PipelineRecentTrack[]>(`/pipeline/recent${q}`);
}

export function ingestPreview(csv_text: string): Promise<IngestPreviewResponse> {
  return request<IngestPreviewResponse>("/pipeline/ingest/preview", {
    method: "POST",
    body: JSON.stringify({ csv_text }),
  });
}

export function ingestCommit(
  csv_text: string,
  opts: { include_duplicates?: boolean; selected_keys?: string[] } = {},
): Promise<Job> {
  return request<Job>("/pipeline/ingest/commit", {
    method: "POST",
    body: JSON.stringify({
      csv_text,
      include_duplicates: opts.include_duplicates ?? false,
      // null lets the backend distinguish "no list provided"
      // (use include_duplicates) from "empty list" (download nothing → 400).
      selected_keys: opts.selected_keys ?? null,
    }),
  });
}

export function triggerPipelineProcess(): Promise<Job> {
  return request<Job>("/pipeline/process", { method: "POST" });
}

export interface ScanResult {
  new: number;
  updated: number;
  unchanged: number;
  errors: number;
}

export function scanLibrary(): Promise<ScanResult> {
  return request<ScanResult>("/pipeline/scan", { method: "POST" });
}

export interface WatchState {
  enabled: boolean;
  interval_seconds: number;
}

export function getWatchState(): Promise<WatchState> {
  return request<WatchState>("/pipeline/watch");
}

export function setWatchState(enabled: boolean): Promise<WatchState> {
  return request<WatchState>("/pipeline/watch", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

export function deleteTrack(id: number): Promise<void> {
  return request<void>(`/tracks/${id}`, { method: "DELETE" });
}

export function getJob(jobId: string): Promise<Job> {
  return request<Job>(`/pipeline/jobs/${jobId}`);
}

export function listJobs(limit = 20): Promise<Job[]> {
  return request<Job[]>(`/pipeline/jobs?limit=${limit}`);
}

// Ableton -------------------------------------------------------------------

export function abletonPlay(): Promise<void> {
  return request<void>("/ableton/play", { method: "POST" });
}

export function abletonStop(): Promise<void> {
  return request<void>("/ableton/stop", { method: "POST" });
}

export function abletonRecord(on: boolean): Promise<{ ok: boolean; recording: boolean }> {
  return request<{ ok: boolean; recording: boolean }>(
    `/ableton/record/${on ? "on" : "off"}`,
    { method: "POST" },
  );
}

export function abletonSetTempo(bpm: number): Promise<void> {
  return request<void>("/ableton/tempo", {
    method: "POST",
    body: JSON.stringify({ bpm }),
  });
}

export function abletonFireClip(track: number, scene: number): Promise<void> {
  return request<void>("/ableton/fire", {
    method: "POST",
    body: JSON.stringify({ track, scene }),
  });
}

// Transport — used by the SceneGrid / per-column rec banners to drive Live
// without the user touching Ableton's UI. See docs/proposals/...

export function abletonFireScene(sceneIndex: number): Promise<void> {
  return request<void>(`/ableton/transport/fire-scene/${sceneIndex}`, {
    method: "POST",
  });
}

export function abletonFireCell(
  trackIndex: number,
  slotIndex: number,
): Promise<void> {
  return request<void>(
    `/ableton/transport/fire-clip/${trackIndex}/${slotIndex}`,
    { method: "POST" },
  );
}

export function abletonStopTrack(trackIndex: number): Promise<void> {
  return request<void>(`/ableton/transport/stop-track/${trackIndex}`, {
    method: "POST",
  });
}

export function abletonStopAllClips(): Promise<void> {
  return request<void>("/ableton/transport/stop-all", { method: "POST" });
}

export function abletonStopCell(
  trackIndex: number,
  slotIndex: number,
): Promise<void> {
  return request<void>(
    `/ableton/transport/stop-cell/${trackIndex}/${slotIndex}`,
    { method: "POST" },
  );
}

export function abletonStopScene(sceneIndex: number): Promise<void> {
  return request<void>(`/ableton/transport/stop-scene/${sceneIndex}`, {
    method: "POST",
  });
}

export function abletonSoloTrack(
  trackIndex: number,
  soloed: boolean,
): Promise<void> {
  return request<void>(
    `/ableton/transport/solo-track/${trackIndex}?soloed=${soloed}`,
    { method: "POST" },
  );
}

export function abletonSeekClip(
  trackIndex: number,
  slotIndex: number,
  positionBeats: number,
): Promise<void> {
  return request<void>(
    `/ableton/transport/seek/${trackIndex}/${slotIndex}?position=${positionBeats}`,
    { method: "POST" },
  );
}

/**
 * Delete one stem-clip from one (track, slot) in the deck columns. Stops
 * the clip if it's playing, then removes the clip from the slot. The
 * slot itself stays — it becomes empty, ready for the next "Load to
 * Live" of that kind.
 */
export function abletonDeleteCell(
  trackIndex: number,
  slotIndex: number,
): Promise<{ ok: boolean; kind?: string; removed_track_id?: number | null }> {
  return request(`/ableton/decks/cell/${trackIndex}/${slotIndex}`, {
    method: "DELETE",
  });
}

export function abletonPreview(body: PreviewRequest): Promise<PreviewResult> {
  return request<PreviewResult>("/ableton/preview", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function abletonPreviewStop(): Promise<PreviewResult> {
  return request<PreviewResult>("/ableton/preview/stop", { method: "POST" });
}

export function abletonSetVolume(
  track: number,
  volume: number,
): Promise<void> {
  return request<void>("/ableton/volume", {
    method: "POST",
    body: JSON.stringify({ track, volume }),
  });
}

export function abletonGetState(): Promise<AbletonState> {
  return request<AbletonState>("/ableton/state");
}

export function abletonDeckMap(): Promise<DeckMap> {
  return request<DeckMap>("/ableton/decks");
}

export function abletonResetDecks(): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/ableton/decks/reset", { method: "POST" });
}

export interface CleanDecksResult {
  ok: boolean;
  deleted: number;
  indices?: number[];
  warning?: string;
}

export function abletonCleanDecks(): Promise<CleanDecksResult> {
  return request<CleanDecksResult>("/ableton/decks/clean", { method: "POST" });
}

export interface ResyncDecksResult {
  ok: boolean;
  scanned: number;
  adopted: number;
  unmatched: Array<{ scene_index: number; kind: string; clip_name: string; tried_title: string }>;
}

export function abletonResyncDecks(): Promise<ResyncDecksResult> {
  return request<ResyncDecksResult>("/ableton/decks/resync", { method: "POST" });
}

// "Push to Live" — drops a (track, stems) bundle into Live's clip grid
// over OSC, using the patched-AbletonOSC ``create_audio_clip`` we ship in
// our fork. No Finder drag needed; stems land directly in the right deck
// columns on a free scene. ``stems_loaded`` reports how many of the 4
// stems made it (less than 4 = check ``warnings``).
export interface LoadTrackResult {
  ok: boolean;
  scene_index: number;
  track_indices: Record<string, number>;
  /** How many of 4 stems landed in Live. 4 = fully auto-loaded, ready to fire. */
  stems_loaded: number;
  message: string | null;
  warnings: string[];
}

export function pushTrackToLive(
  trackId: number,
  options: {
    includeStems?: boolean;
    sceneIndex?: number;
    /** Subset of stem kinds to load. ``undefined`` = whole song (all 4). */
    kinds?: string[];
  } = {},
): Promise<LoadTrackResult> {
  return request<LoadTrackResult>("/ableton/load-track", {
    method: "POST",
    body: JSON.stringify({
      track_id: trackId,
      include_stems: options.includeStems ?? true,
      scene_index: options.sceneIndex ?? null,
      kinds: options.kinds ?? null,
    }),
  });
}

// Live Set export (.als) ---------------------------------------------------
export interface AlsExportResult {
  ok: boolean;
  out_path: string;
  size_bytes: number;
  track_count: number;
  locator_count: number;
}

/** Generate a .als (Ableton Live Set) file for a track on the server. */
export function exportAls(
  trackId: number,
  outPath?: string,
): Promise<AlsExportResult> {
  return request<AlsExportResult>(`/tracks/${trackId}/als`, {
    method: "POST",
    body: JSON.stringify(outPath ? { out_path: outPath } : {}),
  });
}

// Files --------------------------------------------------------------------

export function revealPath(path: string): Promise<{ ok: true }> {
  return request<{ ok: true }>("/files/reveal", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

/** Copy text to the clipboard. Works in modern browsers + Safari on iPad. */
export async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Fallback for older browsers / non-secure contexts.
  const el = document.createElement("textarea");
  el.value = text;
  el.style.position = "fixed";
  el.style.opacity = "0";
  document.body.appendChild(el);
  el.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(el);
  }
}
