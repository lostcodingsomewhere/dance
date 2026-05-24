import { CommandBar } from "./components/CommandBar";
import { MasterStrip } from "./components/MasterStrip";
import { SetRail } from "./components/SetRail";
import { StackMigrationPrompt } from "./components/StackMigrationPrompt";
import { useAppStore } from "./store";
import { Booth } from "./views/Booth";
import { PipelineOps } from "./views/PipelineOps";
import { SetEditor } from "./views/SetEditor";

export function App() {
  const view = useAppStore((s) => s.currentView);
  // Rail is in-flow when open so the SceneGrid + rec banners shrink to
  // make room rather than getting covered. The closed-state edge pill
  // stays fixed-position (rendered by SetRail itself) and overlays.
  // SetEditor manages its own layout; the rail hides there.
  const railOpen = useAppStore((s) => s.setRailOpen);
  const railVisible = railOpen && view !== "set";
  return (
    <div className="h-full w-full flex flex-col bg-neutral-950 text-neutral-100">
      <MasterStrip />
      <div className="flex-1 flex min-h-0">
        <main className="flex-1 flex flex-col min-h-0">
          {view === "booth" && <Booth />}
          {view === "pipeline" && <PipelineOps />}
          {view === "set" && <SetEditor />}
        </main>
        {railVisible && <div className="w-80 shrink-0" aria-hidden />}
      </div>
      <CommandBar />
      <SetRail />
      <StackMigrationPrompt />
    </div>
  );
}
