export interface SectionTone {
  head: string;
  pole: string;
}

const INTRO_OUTRO_TONE: SectionTone = {
  head: "bg-violet-400 text-neutral-950",
  pole: "border-violet-400",
};
const NEUTRAL_TONE: SectionTone = {
  head: "bg-neutral-400 text-neutral-950",
  pole: "border-neutral-400",
};
const SECTION_TONES: Record<string, SectionTone> = {
  intro: INTRO_OUTRO_TONE,
  outro: INTRO_OUTRO_TONE,
  buildup: { head: "bg-yellow-400 text-neutral-950", pole: "border-yellow-400" },
  drop: { head: "bg-red-500 text-white", pole: "border-red-500" },
  breakdown: { head: "bg-cyan-400 text-neutral-950", pole: "border-cyan-400" },
};

export function sectionTone(sectionLabel: string | null | undefined): SectionTone {
  return SECTION_TONES[sectionLabel ?? ""] ?? NEUTRAL_TONE;
}
