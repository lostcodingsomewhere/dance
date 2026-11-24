import type { SeenRequest } from "./boothHarness";

export const API = "/api/v1";
export const ROLES = ["drums", "bass", "vocals", "other", "song"] as const;
export type Role = (typeof ROLES)[number];
export type Plan = Record<Role, number[]>;

export interface Song {
  track_id: number;
  title: string;
}

interface StoredSet {
  id: number;
  name: string;
  active: boolean;
  plan: Plan;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function emptyPlan(): Plan {
  return { drums: [], bass: [], vocals: [], other: [], song: [] };
}

export function rec(song: Song, score = 0.82) {
  return {
    track_id: song.track_id,
    stem_file_id: null,
    track_title: song.title,
    track_artist: "Test Artist",
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score,
    score_breakdown: score > 0 ? { embedding: 0.9 } : {},
    reasons: score > 0 ? ["close vibe"] : [],
  };
}

export function refusalSaying(detail: string, status: number): Response {
  return json({ detail }, status);
}

export class SetsOnTheServer {
  public recs: (role: Role, setId: number) => unknown[] = () => [];
  public unreachableFor: (request: SeenRequest) => boolean = () => false;
  public refuses: (request: SeenRequest) => boolean = () => false;
  public refusal: () => Response = () => new Response("Internal Server Error", { status: 500 });
  private readonly sets: StoredSet[] = [];
  private readonly titles = new Map<number, string>();
  private nextId = 1;

  constructor(library: Song[]) {
    for (const song of library) this.titles.set(song.track_id, song.title);
  }

  has(name: string, plan: Partial<Plan> = {}, active = false): number {
    const stored = { id: this.nextId++, name, active, plan: { ...emptyPlan(), ...plan } };
    this.sets.push(stored);
    return stored.id;
  }

  planOf(setId: number): Plan {
    return this.stored(setId).plan;
  }

  idOf(name: string): number | undefined {
    return this.sets.find((set) => set.name === name)?.id;
  }

  activeName(): string | null {
    return this.sets.find((set) => set.active)?.name ?? null;
  }

  answer = (request: SeenRequest): unknown => {
    if (this.unreachableFor(request)) throw new TypeError("Failed to fetch");
    if (this.refuses(request)) return this.refusal();
    const { method, path, query, body } = request;

    if (path === `${API}/sets`) {
      if (method !== "POST") return this.sets.map((set) => this.summary(set));
      const id = this.has((body as { name: string }).name);
      return this.detail(this.stored(id));
    }
    if (path === `${API}/sets/active`) {
      const active = this.sets.find((set) => set.active);
      return active ? this.detail(active) : json({ detail: "no active set" }, 404);
    }

    const route = path.match(/^\/api\/v1\/sets\/(\d+)(\/.*)?$/);
    if (!route) return undefined;
    const set = this.sets.find((candidate) => candidate.id === Number(route[1]));
    if (!set) return json({ detail: "set not found" }, 404);
    const rest = route[2] ?? "";

    if (rest === "" && method === "DELETE") {
      this.sets.splice(this.sets.indexOf(set), 1);
      return new Response(null, { status: 204 });
    }
    if (rest === "") return this.detail(set);
    if (rest === "/activate") {
      for (const other of this.sets) other.active = other === set;
      return this.detail(set);
    }
    if (rest === "/plan" && method === "PUT") {
      set.plan = { ...emptyPlan(), ...(body as { queues: Partial<Plan> }).queues };
      return this.planOut(set);
    }
    if (rest === "/plan") return this.planOut(set);
    if (rest === "/plan/append") {
      const added = body as { role: Role; track_id: number };
      if (!set.plan[added.role].includes(added.track_id)) set.plan[added.role].push(added.track_id);
      return this.planOut(set);
    }
    if (rest === "/plan-recs") {
      const role = new URLSearchParams(query).get("role") as Role;
      return { column: role === "song" ? "mix" : role, combo_size: 0, recs: this.recs(role, set.id) };
    }
    return undefined;
  };

  private stored(setId: number): StoredSet {
    const found = this.sets.find((set) => set.id === setId);
    if (!found) throw new Error(`the test server has no set ${setId}`);
    return found;
  }

  private summary(set: StoredSet) {
    return {
      id: set.id,
      name: set.name,
      notes: null,
      is_active: set.active,
      track_count: ROLES.reduce((picks, role) => picks + set.plan[role].length, 0),
      created_at: "2026-11-01T00:00:00",
      updated_at: "2026-11-01T00:00:00",
    };
  }

  private detail(set: StoredSet) {
    const { track_count: _picks, ...summary } = this.summary(set);
    return { ...summary, tracks: [] };
  }

  private planOut(set: StoredSet) {
    const item = (track_id: number) => ({
      track_id,
      title: this.titles.get(track_id) ?? null,
      artist: "Test Artist",
      key_camelot: "8A",
      bpm: 128,
      floor_energy: 6,
    });
    return {
      set_id: set.id,
      queues: Object.fromEntries(ROLES.map((role) => [role, set.plan[role].map(item)])),
    };
  }
}

export function holdAnswersTo(method: string, path: string): () => void {
  const answering = globalThis.fetch;
  let release = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const response = await answering(input, init);
    const asked = (typeof input === "string" ? input : input.toString()).split("?")[0];
    if ((init.method ?? "GET") === method && asked === path) await held;
    return response;
  }) as typeof fetch;
  return release;
}
