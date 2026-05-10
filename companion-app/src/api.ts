import type {
  AbletonState,
  AddPlayBody,
  DjSession,
  Recommendation,
  RecommendRequest,
  Region,
  StemFile,
  Track,
  TrackFilters,
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

export function getRegions(trackId: number): Promise<Region[]> {
  return request<Region[]>(`/tracks/${trackId}/regions`);
}

export function getStems(trackId: number): Promise<StemFile[]> {
  return request<StemFile[]>(`/tracks/${trackId}/stems`);
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

// Ableton -------------------------------------------------------------------

export function abletonPlay(): Promise<void> {
  return request<void>("/ableton/play", { method: "POST" });
}

export function abletonStop(): Promise<void> {
  return request<void>("/ableton/stop", { method: "POST" });
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
