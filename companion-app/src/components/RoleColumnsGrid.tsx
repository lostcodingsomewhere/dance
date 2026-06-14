import { PLAN_ROLES } from "../types";
import { ColumnHeader, PlanBand, RecsBand } from "./RoleColumn";

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
  // Rendered as horizontal BANDS rather than five independent columns:
  //   [headers] · [PLAN band] · divider · [RECS band]
  // The PLAN band is a 5-col grid with ``items-stretch``, so every plan zone
  // (filled card or empty "nothing queued") is the height of the tallest —
  // an uneven plan no longer pushes some columns' Recs below the others. The
  // RECS band is a separate grid with ``items-start``, so a column with fewer
  // recs keeps its own height (no dead space). The divider gives a clear seam
  // between what you've ADDED (plan) and what's RECOMMENDED (recs).
  const cols = "grid grid-cols-5 gap-x-2 min-w-0";
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className={cols}>
        {PLAN_ROLES.map((role) => (
          <ColumnHeader key={role} role={role} />
        ))}
      </div>

      {setId != null && (
        <>
          <div className={`${cols} items-stretch`}>
            {PLAN_ROLES.map((role) => (
              <PlanBand key={role} setId={setId} role={role} mode={mode} />
            ))}
          </div>
          {/* seam between added (plan) and recommended (recs) */}
          <div className="mt-1 border-t border-neutral-800" />
        </>
      )}

      <div className={`${cols} items-start`}>
        {PLAN_ROLES.map((role) => (
          <RecsBand key={role} setId={setId} role={role} mode={mode} />
        ))}
      </div>
    </div>
  );
}
