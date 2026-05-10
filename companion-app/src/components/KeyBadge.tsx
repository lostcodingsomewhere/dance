// Renders a Camelot key (e.g. "8A", "11B") as a colored badge.
// Camelot wheel positions 1..12 map to hues around the wheel.

const HUES_1_TO_12 = [
  165, 175, 195, 215, 245, 275, 300, 325, 350, 25, 50, 75,
];

function parseCamelot(key: string): { num: number; letter: "A" | "B" } | null {
  const m = /^(\d{1,2})([AB])$/.exec(key.trim().toUpperCase());
  if (!m) return null;
  const num = Number(m[1]);
  if (num < 1 || num > 12) return null;
  return { num, letter: m[2] as "A" | "B" };
}

export function KeyBadge({
  keyCamelot,
  size = "md",
}: {
  keyCamelot: string | null | undefined;
  size?: "sm" | "md" | "lg";
}) {
  if (!keyCamelot) {
    return (
      <span
        className="inline-flex items-center justify-center rounded-full bg-neutral-800 text-neutral-500 font-mono"
        style={sizeStyle(size)}
      >
        --
      </span>
    );
  }
  const parsed = parseCamelot(keyCamelot);
  if (!parsed) {
    return (
      <span
        className="inline-flex items-center justify-center rounded-full bg-neutral-700 text-neutral-100 font-mono"
        style={sizeStyle(size)}
      >
        {keyCamelot}
      </span>
    );
  }
  const hue = HUES_1_TO_12[parsed.num - 1];
  // A = minor (darker), B = major (lighter).
  const lightness = parsed.letter === "B" ? 60 : 45;
  const bg = `hsl(${hue} 65% ${lightness}%)`;
  const fg = parsed.letter === "B" ? "#0a0a0a" : "#ffffff";
  return (
    <span
      className="inline-flex items-center justify-center rounded-full font-mono font-bold tracking-tight"
      style={{ ...sizeStyle(size), background: bg, color: fg }}
    >
      {keyCamelot}
    </span>
  );
}

function sizeStyle(size: "sm" | "md" | "lg"): React.CSSProperties {
  if (size === "lg") {
    return {
      width: 72,
      height: 72,
      fontSize: 28,
    };
  }
  if (size === "sm") {
    return { width: 36, height: 36, fontSize: 13 };
  }
  return { width: 52, height: 52, fontSize: 18 };
}
