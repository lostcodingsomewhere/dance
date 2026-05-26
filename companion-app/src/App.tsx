import { CommandBar } from "./components/CommandBar";
import { MasterStrip } from "./components/MasterStrip";
import { SetRail } from "./components/SetRail";
import { StackMigrationPrompt } from "./components/StackMigrationPrompt";
import { useAppStore } from "./store";
import { Booth } from "./views/Booth";
import { PipelineOps } from "./views/PipelineOps";
import { SetEditor } from "./views/SetEditor";

/**
 * Top-level shell. The SetRail's layout is *view-aware*:
 *
 *   - In **Booth**: rail renders as an inline sibling of the main column.
 *     The MasterStrip is inside that main column (not a top-level row), so
 *     when the rail is pinned/open the *nav itself* slides over too —
 *     matches the user's explicit "make sure nav also is pushed with it
 *     over" requirement. Default pinned-open per ``store.setRailPinned``.
 *   - In **Pipeline**: rail can still toggle via ⌘\ — when open, same
 *     inline-push behavior; when closed, the closed-pill tab on the right
 *     edge stays fixed-position.
 *   - In **Set**: the SetEditor view renders the rail content full-pane
 *     itself, so we don't render the drawer/tab here.
 */
export function App() {
  const view = useAppStore((s) => s.currentView);
  const railOpen = useAppStore((s) => s.setRailOpen);
  const railPinned = useAppStore((s) => s.setRailPinned);

  // The rail is "shown" (drawer mode) when explicitly open, or when in
  // Booth + pinned. Booth's pinned default means a fresh load already
  // shows the rail without the user toggling it.
  const railShown = view !== "set" && (railOpen || (view === "booth" && railPinned));

  return (
    <div className="h-full w-full flex bg-neutral-950 text-neutral-100">
      {/* Main column — MasterStrip + the active view's content. Sits
          alongside the rail, so the rail's presence makes both the nav
          and the body shrink together rather than the rail overlaying
          the nav. ``min-w-0`` so flex-children with long truncating text
          don't push the whole row sideways. */}
      <div className="flex-1 min-w-0 flex flex-col">
        <MasterStrip />
        <main className="flex-1 flex flex-col min-h-0">
          {view === "booth" && <Booth />}
          {view === "pipeline" && <PipelineOps />}
          {view === "set" && <SetEditor />}
        </main>
      </div>
      {/* SetRail renders here as a flex sibling when shown. When hidden in
          a view where it's still available (Pipeline, Booth-unpinned), it
          renders a closed-pill tab in fixed position (handled inside
          SetRail itself). */}
      <SetRail inline={railShown} />
      <CommandBar />
      <StackMigrationPrompt />
    </div>
  );
}
