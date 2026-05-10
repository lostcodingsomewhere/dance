import { TopBar } from "./components/TopBar";
import { useAppStore } from "./store";
import { Library } from "./views/Library";
import { NowPlaying } from "./views/NowPlaying";
import { SessionHistory } from "./views/SessionHistory";
import { UpNext } from "./views/UpNext";

export function App() {
  const view = useAppStore((s) => s.currentView);
  return (
    <div className="h-full w-full flex flex-col bg-neutral-950 text-neutral-100">
      <TopBar />
      <main className="flex-1 flex flex-col min-h-0">
        {view === "now" && <NowPlaying />}
        {view === "next" && <UpNext />}
        {view === "library" && <Library />}
        {view === "session" && <SessionHistory />}
      </main>
    </div>
  );
}
