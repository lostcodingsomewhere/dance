import { CommandBar } from "./components/CommandBar";
import { MasterStrip } from "./components/MasterStrip";
import { useAppStore } from "./store";
import { Booth } from "./views/Booth";
import { Crate } from "./views/Crate";
import { PipelineOps } from "./views/PipelineOps";

export function App() {
  const view = useAppStore((s) => s.currentView);
  return (
    <div className="h-full w-full flex flex-col bg-neutral-950 text-neutral-100">
      <MasterStrip />
      <main className="flex-1 flex flex-col min-h-0">
        {view === "booth" && <Booth />}
        {view === "crate" && <Crate />}
        {view === "pipeline" && <PipelineOps />}
      </main>
      <CommandBar />
    </div>
  );
}
