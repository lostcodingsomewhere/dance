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
  return (
    <div className="h-full w-full flex flex-col bg-neutral-950 text-neutral-100">
      <MasterStrip />
      <main className="flex-1 flex flex-col min-h-0">
        {view === "booth" && <Booth />}
        {view === "pipeline" && <PipelineOps />}
        {view === "set" && <SetEditor />}
      </main>
      <CommandBar />
      <SetRail />
      <StackMigrationPrompt />
    </div>
  );
}
