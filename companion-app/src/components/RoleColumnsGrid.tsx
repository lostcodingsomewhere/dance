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
  return (
    <div className="grid grid-cols-5 gap-2 min-w-0">
      {PLAN_ROLES.map((role) => (
        <RoleColumn key={role} setId={setId} role={role} mode={mode} />
      ))}
    </div>
  );
}
