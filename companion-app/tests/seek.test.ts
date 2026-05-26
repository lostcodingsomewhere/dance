import { describe, expect, it } from "vitest";
import { ratioToBeats, snapRatioToSection } from "../src/lib/seek";
import type { Region } from "../src/types";

const section = (id: number, ms: number): Region => ({
  id,
  region_type: "section",
  position_ms: ms,
  length_ms: 30_000,
  section_label: "drop",
  bar_count: 8,
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
      { id: 99, region_type: "cue", position_ms: 61_200, length_ms: null },
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
