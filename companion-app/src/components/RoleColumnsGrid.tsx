import { PLAN_ROLES } from "../types";
import { RoleColumn } from "./RoleColumn";

/**
 * The one grid the app is built around: five role columns, each with your
 * queued plan picks on top and recs below. Used in the Booth (mode="live" —
 * recs tail off what's playing) and the Set page (mode="plan" — recs scored
 * against the rest of the plan). ``setId`` may be null in the Booth when no
 * set is active (then it's just the live recs, no plan zone).
 */
export function RoleColumnsGrid({
  setId,
  mode,
}: {
  setId: number | null;
  mode: "plan" | "live";
}) {
  // Share row tracks across the five columns via CSS subgrid so the
  // header → plan-label → plan-zone → recs-label rows line up no matter how
  // many picks each column has queued. The plan-zone track auto-sizes to the
  // TALLEST plan zone, which stretches the empty "nothing queued" box to match
  // a filled card — so an uneven plan (1 column queued, 4 empty) no longer
  // pushes some columns' Recs below the others. The recs LIST sits in a
  // per-column implicit row below, so columns with more recs stay independent
  // (no forced equal height / dead space).
  const sharedRows = setId != null ? 4 : 2; // header,[plan-label,plan-zone,]recs-label
  return (
    <div
      className="grid grid-cols-5 gap-x-2 gap-y-1 min-w-0"
      style={{ gridTemplateRows: `repeat(${sharedRows}, auto)` }}
    >
      {PLAN_ROLES.map((role) => (
        <RoleColumn
          key={role}
          setId={setId}
          role={role}
          mode={mode}
          sharedRows={sharedRows}
        />
      ))}
    </div>
  );
}
