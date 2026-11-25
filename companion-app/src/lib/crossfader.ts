import { sideOf } from "./roles";

// A hardware crossfader pushed to the end may stop just short of ±1.
const FULLY_ACROSS = 0.95;

export function isHeardThroughTheCrossfader(deckKind: string, crossfader: number | null | undefined): boolean {
  const side = sideOf(deckKind);
  if (side == null || crossfader == null) return true;
  return side === "a" ? crossfader < FULLY_ACROSS : crossfader > -FULLY_ACROSS;
}
