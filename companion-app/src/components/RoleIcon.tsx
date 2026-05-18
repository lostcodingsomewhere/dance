/**
 * Monochrome line-art icons for the 5 stem roles. Uses ``currentColor`` so
 * the icon takes its color from the wrapper's text color (or className).
 *
 * Each icon is a simple recognisable glyph that reads at 12-16px sizes:
 *   drums  → a snare-drum cylinder
 *   bass   → a sine-ish low-frequency wave
 *   vocals → a microphone
 *   other  → an eighth-note (melody / harmony stem)
 *   mix    → a record disc (the whole song)
 *
 * If we ever want richer icons (lucide, heroicons), swap the path data
 * here — the rest of the UI doesn't care how the icon is drawn.
 */

import type { StemRole } from "../lib/roles";

interface RoleIconProps {
  role: StemRole | string;
  className?: string;
  size?: number;
}

export function RoleIcon({ role, className, size = 14 }: RoleIconProps) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    className,
    "aria-hidden": true,
  };

  switch (role) {
    case "drums":
      return (
        <svg {...common}>
          {/* drum head ellipse + side wall */}
          <ellipse cx="12" cy="7" rx="8" ry="3" />
          <path d="M4 7 v10 c0 1.6 3.6 3 8 3 s8 -1.4 8 -3 V7" />
          <line x1="4" y1="17" x2="20" y2="17" />
        </svg>
      );
    case "bass":
      return (
        <svg {...common}>
          {/* low-frequency wave */}
          <path d="M2 12 Q 5 4 8 12 T 14 12 T 20 12 T 22 12" />
        </svg>
      );
    case "vocals":
      return (
        <svg {...common}>
          {/* microphone capsule + stand */}
          <rect x="9" y="3" width="6" height="11" rx="3" />
          <path d="M5 11 a7 7 0 0 0 14 0" />
          <line x1="12" y1="18" x2="12" y2="21" />
          <line x1="9" y1="21" x2="15" y2="21" />
        </svg>
      );
    case "other":
      return (
        // melody — an eighth-note with a beam
        <svg {...common}>
          <circle cx="6" cy="18" r="3" />
          <path d="M9 18 V5 L20 7 V16" />
          <circle cx="17" cy="16" r="3" />
        </svg>
      );
    case "mix":
      return (
        // song — a record disc
        <svg {...common} strokeWidth={1.6}>
          <circle cx="12" cy="12" r="9" />
          <circle cx="12" cy="12" r="3" />
          <circle cx="12" cy="12" r="0.7" fill="currentColor" />
        </svg>
      );
    default:
      return null;
  }
}
