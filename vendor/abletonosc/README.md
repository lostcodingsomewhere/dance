# AbletonOSC patches

Dance talks to Live through [AbletonOSC](https://github.com/ideoforms/AbletonOSC), a Remote Script installed at
`~/Music/Ableton/User Library/Remote Scripts/AbletonOSC`. Stock AbletonOSC is missing a few handlers dance sends to;
without them Live drops the message and the bridge silently does nothing. These patches add them. They are pinned
to the upstream commit in [`UPSTREAM`](UPSTREAM) so every Mac runs the same fork.

| Patch | Adds | Used by |
|---|---|---|
| `0001` | `/live/clip_slot/create_audio_clip` (Live 12.0.5+) | auto-loading stems into deck slots |
| `0002` | fixes the `clip_slot_callback` logging call that raised on every clip-slot method | (log noise; dropped replies) |
| `0003` | `/live/track/{get,set}/crossfade_assign` | routing app-loaded decks to crossfader A/B |
| `0004` | `/live/song/{get,set,start_listen,stop_listen}/crossfader` | mirroring + driving the master crossfader |
| `0005` | `/live/clip/get/warp_marker_times`, `/live/clip/{add,move,remove}_warp_marker` | warp audit/repair (`scripts/warp_probe.py`, `scripts/grid_diff.py --live`) |

## Install / check

```bash
python scripts/abletonosc_patch.py --check   # report what's applied; changes nothing
python scripts/abletonosc_patch.py           # back up, then apply what's missing
```

Then **fully quit and reopen Live** — toggling the Control Surface off/on isn't enough (Live keeps the modules in
`sys.modules`). `dance doctor` reports any address dance sends that the installed script doesn't handle.

The script applies the patches as working-tree changes (`git apply`), never commits in the AbletonOSC clone, and
refuses to touch anything if a patch neither applies cleanly nor is already applied. The backup goes to
`~/.dance/backups/abletonosc-<timestamp>/` — outside Remote Scripts, where Live would try to load it as a second
script.

## Updating a patch

Clone the install (`git clone "<install dir>" /tmp/aosc`), check out the pinned commit, `git am` the series, edit,
commit, and re-export with `git format-patch --no-signature -o vendor/abletonosc/patches <commit>..HEAD`.
