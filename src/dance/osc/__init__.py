"""OSC bridge to Ableton Live (via the AbletonOSC plugin).

Public surface:
    AbletonBridge     — combined send/receive client; the thing the FastAPI
                        backend instantiates.
    AbletonOSCClient  — send-only.
    AbletonOSCListener — receive-only (subscribe to state pushes).
    AbletonState      — most recent observed Live state (tempo, beat, clips).

Install AbletonOSC into Ableton Live separately — see docs/abletonosc_setup.md.
"""

from dance.osc.bridge import AbletonBridge, AbletonState
from dance.osc.client import AbletonOSCClient
from dance.osc.listener import AbletonOSCListener

__all__ = [
    "AbletonBridge",
    "AbletonOSCClient",
    "AbletonOSCListener",
    "AbletonState",
]
