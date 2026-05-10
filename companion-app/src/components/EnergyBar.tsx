// Visual 1-10 energy bar: 10 cells, fill = floor_energy, color shifts green→yellow→red.

function cellColor(idx: number): string {
  // idx is 1-based 1..10.
  if (idx <= 3) return "bg-emerald-500";
  if (idx <= 6) return "bg-lime-400";
  if (idx <= 8) return "bg-amber-400";
  return "bg-red-500";
}

export function EnergyBar({
  energy,
  size = "md",
}: {
  energy: number | null | undefined;
  size?: "sm" | "md" | "lg";
}) {
  const filled = Math.max(0, Math.min(10, energy ?? 0));
  const cellH = size === "lg" ? "h-7" : size === "sm" ? "h-2.5" : "h-4";
  const gap = size === "sm" ? "gap-[2px]" : "gap-1";
  return (
    <div className={`flex ${gap} items-end`}>
      {Array.from({ length: 10 }, (_, i) => {
        const idx = i + 1;
        const on = idx <= filled;
        const w = size === "lg" ? "w-5" : size === "sm" ? "w-2" : "w-3";
        return (
          <div
            key={idx}
            className={`${cellH} ${w} rounded-sm ${
              on ? cellColor(idx) : "bg-neutral-800"
            }`}
          />
        );
      })}
      {size !== "sm" && (
        <span className="ml-2 font-mono text-neutral-400 text-sm">
          {energy == null ? "--" : `${filled}/10`}
        </span>
      )}
    </div>
  );
}
