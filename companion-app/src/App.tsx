import { MutationCache, QueryClient, useQueryClient, type Mutation } from "@tanstack/react-query";
import { Component, type ReactNode } from "react";

import { serverMessage } from "./api";
import { CommandBar } from "./components/CommandBar";
import { LoadWarnings } from "./components/LoadWarnings";
import { MasterStrip } from "./components/MasterStrip";
import { StackMigrationPrompt } from "./components/StackMigrationPrompt";
import { store, useAppStore } from "./store";
import { Booth } from "./views/Booth";
import { PipelineOps } from "./views/PipelineOps";
import { SetEditor } from "./views/SetEditor";

const UNMATCHED_CLIPS_NAMED = 3;

const BACKEND_DID_NOT_ANSWER = "The Dance backend did not answer. Check it is still running.";

type AnyMutation = Mutation<unknown, unknown, unknown>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function showsItsOwnFailure(mutation: AnyMutation): boolean {
  return mutation.meta?.showsOwnFailure === true;
}

function whatWasAttempted(mutation: AnyMutation, variables: unknown): string | null {
  const attempt = mutation.meta?.attempt;
  if (typeof attempt === "string") return attempt;
  if (typeof attempt !== "function") return null;
  // A throw here would turn a command that worked into one that failed.
  try {
    return String(attempt(variables));
  } catch {
    return null;
  }
}

function whyItFailed(error: Error): string {
  // fetch rejects with a TypeError when no answer came back at all.
  return error instanceof TypeError ? BACKEND_DID_NOT_ANSWER : serverMessage(error);
}

function whyItWasRefused(answer: unknown): string | null {
  if (!isRecord(answer) || answer.ok !== false) return null;
  if (typeof answer.warning === "string" && answer.warning) return answer.warning;
  if (Array.isArray(answer.warnings) && answer.warnings.length > 0) return answer.warnings.join(" ");
  return "It was refused, and no reason was given.";
}

function clipsResyncCouldNotMatch(answer: unknown): string[] {
  const unmatched = isRecord(answer) ? answer.unmatched : null;
  if (!Array.isArray(unmatched)) return [];
  return unmatched.map((clip) => (isRecord(clip) ? String(clip.clip_name) : String(clip)));
}

function reportUnmatchedClips(names: string[]): void {
  if (names.length === 0) return;
  const named = names.slice(0, UNMATCHED_CLIPS_NAMED).join(", ");
  const notNamed = names.length - UNMATCHED_CLIPS_NAMED;
  store.setLoadWarnings(
    `Resync couldn't match ${names.length} ${names.length === 1 ? "clip" : "clips"} to your library`,
    [notNamed > 0 ? `${named} and ${notNamed} more` : named],
  );
}

export function createAppQueryClient(): QueryClient {
  return new QueryClient({
    mutationCache: new MutationCache({
      onError: (error, variables, _onMutateResult, mutation) => {
        if (showsItsOwnFailure(mutation)) return;
        store.reportFailure(whatWasAttempted(mutation, variables), [whyItFailed(error)]);
      },
      onSuccess: (answer, variables, _onMutateResult, mutation) => {
        if (showsItsOwnFailure(mutation)) return;
        const refusal = whyItWasRefused(answer);
        if (refusal) store.reportFailure(whatWasAttempted(mutation, variables), [refusal]);
        reportUnmatchedClips(clipsResyncCouldNotMatch(answer));
      },
    }),
    defaultOptions: {
      queries: {
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  });
}

class ScreenBoundary extends Component<
  { onReload: () => void; children: ReactNode },
  { broke: Error | null }
> {
  state: { broke: Error | null } = { broke: null };

  static getDerivedStateFromError(thrown: unknown): { broke: Error } {
    return { broke: thrown instanceof Error ? thrown : new Error(String(thrown)) };
  }

  reload = (): void => {
    this.props.onReload();
    this.setState({ broke: null });
  };

  render(): ReactNode {
    const { broke } = this.state;
    if (!broke) return this.props.children;
    return (
      <div
        role="alert"
        className="m-4 rounded-md border border-rose-500/50 bg-rose-500/10 px-4 py-3"
      >
        <h2 className="text-sm font-semibold text-rose-100">Something broke on this screen</h2>
        <p className="mt-1 break-words font-mono text-xs text-rose-200/90">{broke.message}</p>
        <p className="mt-2 text-xs text-neutral-300">
          The transport and Panic above still work.
        </p>
        <button
          type="button"
          onClick={this.reload}
          className="mt-3 min-h-[40px] rounded-md bg-rose-600 px-4 text-sm font-semibold text-white hover:bg-rose-500"
        >
          Reload
        </button>
      </div>
    );
  }
}

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
  const queryClient = useQueryClient();

  return (
    <div className="h-full w-full flex flex-col bg-neutral-950 text-neutral-100">
      <MasterStrip />
      {/* Warp / load warnings — directly under the master strip so a bad
          auto-warp is impossible to miss before you fire the deck. */}
      <LoadWarnings />
      <main className="flex-1 flex flex-col min-h-0">
        <ScreenBoundary key={view} onReload={() => void queryClient.resetQueries()}>
          {view === "booth" && <Booth />}
          {view === "pipeline" && <PipelineOps />}
          {view === "set" && <SetEditor />}
        </ScreenBoundary>
      </main>
      <CommandBar />
      <StackMigrationPrompt />
    </div>
  );
}
