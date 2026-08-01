import { CommandBar } from "./components/CommandBar";
import { LoadWarnings } from "./components/LoadWarnings";
import { MasterStrip } from "./components/MasterStrip";
import { StackMigrationPrompt } from "./components/StackMigrationPrompt";
import { useAppStore } from "./store";
import { Booth } from "./views/Booth";
import { PipelineOps } from "./views/PipelineOps";
import { SetEditor } from "./views/SetEditor";

/**
 * Top-level shell. One model, no sidebar:
 *   - **Booth** performs the set (now-playing → Ableton grid → plan grid →
 *     recs that tail off what's playing).
 *   - **Set** authors the set as per-role plan queues (the same rec grid).
 *   - **Pipeline** is ingest/processing status.
 * ⌘K is the universal finder across all of them.
 */
export function App() {
  const view = useAppStore((s) => s.currentView);

  return (
    <div className="h-full w-full flex flex-col bg-neutral-950 text-neutral-100">
      <MasterStrip />
      {/* Warp / load warnings — directly under the master strip so a bad
          auto-warp is impossible to miss before you fire the deck. */}
      <LoadWarnings />
      <main className="flex-1 flex flex-col min-h-0">
        {view === "booth" && <Booth />}
        {view === "pipeline" && <PipelineOps />}
        {view === "set" && <SetEditor />}
      </main>
      <CommandBar />
      <StackMigrationPrompt />
    </div>
  );
}
