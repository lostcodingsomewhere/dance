import { store, useAppStore } from "../store";

export function PinButton({ trackId }: { trackId: number }) {
  const pinned = useAppStore((s) => s.pinnedSeeds.includes(trackId));
  return (
    <button
      type="button"
      onClick={() => (pinned ? store.unpin(trackId) : store.pin(trackId))}
      className={`min-h-[44px] px-4 rounded-lg font-semibold text-sm transition-colors ${
        pinned
          ? "bg-amber-400 text-neutral-950 hover:bg-amber-300"
          : "bg-neutral-800 text-neutral-200 hover:bg-neutral-700"
      }`}
      aria-pressed={pinned}
      title={pinned ? "Unpin from seeds" : "Pin as seed"}
    >
      {pinned ? "Pinned" : "Pin"}
    </button>
  );
}
