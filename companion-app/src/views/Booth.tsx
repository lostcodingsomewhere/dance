import { BoothColumnHeaders } from "../components/BoothColumnHeaders";
import { ColumnRecBanner } from "../components/ColumnRecBanner";
import { Crossfader } from "../components/Crossfader";
import { CueStrip } from "../components/CueStrip";
import { SceneGrid } from "../components/SceneGrid";
import { TwoDeckStrip } from "../components/TwoDeckStrip";
import { useAutoLog } from "../hooks/useAutoLog";
import { useAutoSession } from "../hooks/useAutoSession";

/**
 * The Booth — the only screen you should look at during a set.
 *
 * Live-remixing layout (two-deck rethink — see
 * docs/proposals/two-deck-ui-rethink.md):
 *
 *   ┌─ MasterStrip (BPM · KEY · energy arc · session chip · ⌘K · tabs) ─┐
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ TwoDeckStrip (Deck A / Deck B stacked-stem waveforms + per-deck   │
 *   │   header: ◇ mix reference, PFL, FX, sync)                         │
 *   │ Crossfader (A | B — mirrors APC40 hardware fader)                 │
 *   │ BoothColumnHeaders (8 chips: DRUMS A · DRUMS B · BASS A · … in    │
 *   │   APC40 fader order, with Solo "S" buttons)                       │
 *   │ SceneGrid (8-column APC40 mirror — tap cells, tap rows; ▾ expand) │
 *   │ CueStrip (prelisten — headphones out)                             │
 *   │ Rec banners: 4 source-role feeds, each spanning its A+B columns   │
 *   │   (drums · bass · vocals · other), + a full-width song feed below │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Session play count + end-session lives in the MasterStrip's SessionChip
 * (right side). The Set Rail (⌘\) covers planning + tail-recs. No footer
 * — the SceneGrid is the only thing that should be earning the
 * bottom-of-screen real estate.
 *
 * BoothColumnHeaders → SceneGrid → the role-rec banner row all share the
 * same ``grid-cols-8 gap-1`` grid template, so the 8 stem columns line up
 * vertically — flush to the left edge, in APC40 fader order — and the
 * labels at the top govern everything beneath them. There's no on-screen
 * row-number / scene-launch column; the APC40's hardware scene-launch
 * buttons fire whole scenes.
 *
 * Side effects:
 *   - Auto-creates a session on first Ableton play.
 *   - Auto-logs plays as Ableton fires clips loaded via Load-to-Live.
 */
export function Booth() {
  useAutoSession();
  useAutoLog();

  // Source roles for the rec banner — 4 stem feeds, each spanning its
  // role's two grid columns (A+B). Recs are role-scoped (not side-scoped);
  // each card's ⤒A/⤒B buttons pick the deck at load time. The "mix" (song)
  // feed is a full-width banner below, since the mix is no longer a grid
  // column.
  const recRoles = ["drums", "bass", "vocals", "other"];

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <main className="flex-1 flex flex-col min-h-0 gap-2 px-4 py-3 overflow-y-auto">
        {/* Two-deck Traktor-style "now playing" — replaces ComboStrip.
            One stacked-stem waveform per deck side. */}
        <TwoDeckStrip />
        {/* Crossfader bar — mirrors APC40 hardware crossfader, also
            drag-to-set on screen. Sits visually between the two deck
            panels above and the per-stem grid below. */}
        <Crossfader />
        <BoothColumnHeaders />
        <SceneGrid />
        <CueStrip />
        {/* Recs banner: 4 source-role feeds. Each card spans 2 of the 8
            grid cols below so it sits above its A/B pair. */}
        <div className="grid grid-cols-8 gap-1">
          {recRoles.map((c) => (
            <div key={c} className="col-span-2">
              <ColumnRecBanner column={c} k={5} />
            </div>
          ))}
        </div>
        {/* Song / anchor rec banner — full width below the per-role
            feeds. These are whole-song candidates (load = all 4 stems
            into a deck). */}
        <ColumnRecBanner column="mix" k={5} />
      </main>
    </div>
  );
}
