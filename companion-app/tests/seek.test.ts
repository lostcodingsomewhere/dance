import { describe, expect, it } from "vitest";
import { clipPlayhead, ratioToBeats, snapRatioToSection } from "../src/lib/seek";
import type { Region } from "../src/types";

// Minimal Region factory — fills every required field so the literal
// satisfies the current Region type; the seek logic only reads
// region_type + position_ms.
const region = (over: Partial<Region> & Pick<Region, "id" | "region_type" | "position_ms">): Region => ({
  length_ms: 30_000,
  section_label: null,
  length_bars: null,
  name: null,
  color: null,
  confidence: null,
  source: "test",
  stem_file_id: null,
  ...over,
});

const section = (id: number, ms: number): Region =>
  region({
    id,
    region_type: "section",
    position_ms: ms,
    length_ms: 30_000,
    section_label: "drop",
    length_bars: 8,
  });

describe("snapRatioToSection", () => {
  const durationSec = 240; // 4 min
  const regions: Region[] = [
    section(1, 0),        // 0%
    section(2, 60_000),   // 25%
    section(3, 120_000),  // 50%
  ];

  it("snaps to nearest section within tolerance", () => {
    // 52% click → snap forward to 50% section
    expect(snapRatioToSection(0.52, regions, durationSec)).toBeCloseTo(0.5);
  });

  it("returns exact ratio when no section is within tolerance", () => {
    // 80% click, last section is at 50%, far outside default 8% window
    expect(snapRatioToSection(0.8, regions, durationSec)).toBeCloseTo(0.8);
  });

  it("snaps forward (not just backward) — fixes the click-past-half bug", () => {
    // 23% click is closer to 25% section than 0% section → snap forward
    expect(snapRatioToSection(0.23, regions, durationSec)).toBeCloseTo(0.25);
  });

  it("returns ratio verbatim when regions are empty", () => {
    expect(snapRatioToSection(0.7, [], durationSec)).toBe(0.7);
    expect(snapRatioToSection(0.3, undefined, durationSec)).toBe(0.3);
  });

  it("returns ratio verbatim when duration is zero", () => {
    expect(snapRatioToSection(0.5, regions, 0)).toBe(0.5);
  });

  it("ignores non-section regions", () => {
    const mixed: Region[] = [
      ...regions,
      // Cue at 25.5% — would snap there if cues counted, but they don't.
      region({ id: 99, region_type: "cue", position_ms: 61_200, length_ms: null }),
    ];
    // 26% click → snaps to the 25% *section*, not the cue.
    expect(snapRatioToSection(0.26, mixed, durationSec)).toBeCloseTo(0.25);
  });
});

describe("ratioToBeats", () => {
  it("converts a 0-1 ratio into clip beats given duration + bpm", () => {
    // 4-min track at 120 BPM = 480 beats; halfway = 240 beats
    expect(ratioToBeats(0.5, 240, 120)).toBeCloseTo(240);
  });

  it("returns 0 at ratio 0", () => {
    expect(ratioToBeats(0, 240, 120)).toBe(0);
  });
});

describe("clipPlayhead", () => {
  it("maps a live beat position to the matching 0-1 ratio + seconds", () => {
    // 4-min @ 120 BPM: 240 beats in = 120 s = halfway.
    const v = clipPlayhead(240, 240, 120);
    expect(v?.position).toBeCloseTo(0.5);
    expect(v?.elapsedSec).toBeCloseTo(120);
    expect(v?.remaining).toBeCloseTo(120);
  });

  it("wraps a looped beat position back into range", () => {
    // 600 beats on a 480-beat (4-min @120) clip → 120 beats in → 25%.
    expect(clipPlayhead(600, 240, 120)?.position).toBeCloseTo(0.25);
  });

  it("returns null without a usable duration or bpm", () => {
    expect(clipPlayhead(100, 0, 120)).toBeNull();
    expect(clipPlayhead(100, 240, 0)).toBeNull();
  });

  it("is the exact inverse of ratioToBeats — playhead lands where a click seeks", () => {
    // This is the property that fixes "clickability": the playhead the cue
    // strip draws and the beat a same-ratio click seeks to use one conversion.
    for (const r of [0, 0.1, 0.37, 0.5, 0.99]) {
      const beats = ratioToBeats(r, 240, 128);
      expect(clipPlayhead(beats, 240, 128)?.position).toBeCloseTo(r);
    }
  });
});
