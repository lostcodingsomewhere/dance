import { useState } from "react";
import { copyToClipboard, revealPath } from "../api";

/**
 * "Get this into Ableton" affordances. Drag-and-drop from Finder is the
 * fastest way today; these buttons give the user the path on their clipboard
 * or open Finder positioned on the file so they can drag.
 *
 * When AbletonOSC clip-load lands, the "Push to Live" button will replace
 * the need for these — but keep both around since drag-and-drop is the most
 * reliable workflow.
 */
export function LoadActions({
  path,
  label = "Track",
}: {
  path: string | null | undefined;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!path) return null;

  const copy = async () => {
    try {
      setError(null);
      await copyToClipboard(path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const reveal = async () => {
    try {
      setError(null);
      await revealPath(path);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={copy}
        title={`Copy ${label.toLowerCase()} path`}
        className="h-10 px-3 rounded-md text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 active:scale-95"
      >
        {copied ? "✓ Copied" : "Copy path"}
      </button>
      <button
        type="button"
        onClick={reveal}
        title={`Reveal ${label.toLowerCase()} in Finder`}
        className="h-10 px-3 rounded-md text-xs font-medium bg-neutral-800 hover:bg-neutral-700 text-neutral-200 active:scale-95"
      >
        Reveal
      </button>
      {error && (
        <span className="text-xs text-red-400 ml-1" title={error}>!</span>
      )}
    </div>
  );
}
