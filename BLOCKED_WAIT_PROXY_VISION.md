# Blocked Wait Proxy Vision

Goal: when the main Hermes session is blocked on an interactive/tool wait, Hermes should still feel socially present and steerable instead of looking frozen.

Core idea
- Treat every blocked state as a generic blocked-session substrate.
- The substrate records:
  - wait kind
  - when it started
  - current activity
  - child-agent summaries
  - allowed actions
- Then attach a thin per-wait adapter (clarify, delegate, approval, update, etc.).
- A cheap continuity proxy can be launched on top of that substrate so the user is still talking to “Hermes” while the main run is paused.

Proxy model
- Prefer a cheap local Qwen-style model when configured.
- If no dedicated proxy model is configured, inherit the current Hermes runtime.
- The proxy is not a separate persona; it is the blocked-session face of the main Hermes thread.

Proxy prompt shape
1. Identity / continuity layer
- “You are the blocked-session continuity proxy for the active Hermes session.”
- “You are not a separate assistant from the user’s perspective.”
- “Speak like the main assistant and preserve its goals.”

2. Recent-context layer
- recent assistant style samples
- recent transcript excerpt (bounded)
- current blocked wait snapshot

3. Adapter layer
- dynamic instructions based on the current wait kind
- example: delegate adapter explains the child’s status and what steering/abort means
- example: clarify adapter explains what answer Hermes is waiting for and what `/answer` does

No-config behavior
- If the user tries to use the helper for a wait kind that has no config, Hermes should pause and ask whether to create a default config.
- Timeout: 90 seconds.
- A “yes” response writes a small default adapter config for that wait kind.
- A “no” response leaves the helper disabled and falls back to normal behavior.

Implementation layers
1. Base substrate
- `run_agent.py` activity + wait-state metadata
- generic blocked-wait dispatch path in the gateway

2. Thin adapters
- update prompt wait
- dangerous-command approval wait
- clarify wait
- delegate wait

3. Proxy runtime
- `agent/blocked_wait_proxy.py`
- bounded context excerpt
- continuity prompt
- per-kind adapter instructions
- cheap runtime override from config when available

Current implementation status on this branch
- generic wait-state metadata exists
- blocked-wait dispatch exists
- delegate watchdog / child heartbeat exists
- blocked-session proxy config + runtime exists
- gateway can prompt to enable the proxy when a wait kind lacks config
- gateway can use the proxy for blocked clarify/delegate meta-questions
- normal free-text during delegate waits still falls through as steering/interruption

Endgame
- make the proxy/profile system fully reusable across all blocked waits
- allow named profiles / launch presets for reviewer, delegate supervisor, or continuity helper roles
- let tools specify only their adapter instructions while the substrate + launch mechanism remain generic
