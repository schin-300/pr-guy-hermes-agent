# Agent Browser v1 plan

Goal
- Ship a native Hermes CLI agent browser that opens with Ctrl+B and works immediately after a normal Hermes update.
- The browser must use Hermes profiles as the source of truth and feel like an identity-rich profile browser rather than a plain list.

User workflow
1. User is in `hermes chat`.
2. User presses Ctrl+B.
3. Hermes opens an in-terminal browser view of all profiles.
4. The current profile immediately starts a background reflective status generation pass.
5. Cards refresh automatically every ~2s from local files/runtime state.
6. User can move selection with arrows / j / k, use Enter to switch active profile, and Esc / Ctrl+B to close.

v1 scope
- Native prompt_toolkit browser surface inside the existing Hermes CLI.
- Ctrl+B opens/closes the browser when voice mode is not active.
- Profile cards show: profile name, active marker, wired presence, model/provider, skin/theme identity, current status summary, feeling/reflection snippet, updated time.
- Background status generation for the current profile using Hermes-native auxiliary LLM plumbing.
- Status persistence per profile with current + history files.
- Lightweight live refresh via local polling / heartbeat, no AI loop for refresh.
- Current session/profile switching via Enter on a selected card.

Non-goals for v1
- No GUI/web app.
- No image-rendering dependency or kitty-only path.
- No cross-process remote orchestration beyond local profile state discovery.
- No README rewrite.

System shape
1. `hermes_cli/agent_browser.py`
   - profile discovery + card snapshot assembly
   - presence heartbeat helpers
   - status file read/write helpers
   - background reflective status generation
   - terminal-safe card rendering helpers
2. `cli.py`
   - browser state on `HermesCLI`
   - Ctrl+B keybinding routing: browser toggle when voice mode is off; existing voice record handler remains for voice mode
   - extra browser widget in TUI layout
   - background refresh / invalidate hooks
   - lifecycle hooks for presence heartbeat start/stop
3. `hermes_cli/profiles.py`
   - enrich `ProfileInfo` with skin/last-activity/browser-facing fields needed by cards
4. tests
   - browser snapshot/status persistence tests
   - Ctrl+B/browser keybinding tests
   - profile switch / selection behavior tests
   - presence heartbeat tests

Persistence layout
- Per-profile root: `<profile_home>/workspace/agent-browser/`
- Current status: `current.json`
- Status history: `history/<random_id>.json`
- Presence heartbeats: `presence/<pid>.json`

Status generation shape
- Reuse Hermes auxiliary LLM path similar to title generation.
- Prompt includes:
  - recent conversation slice
  - session title if any
  - profile memory snippets when available
  - instructions to output strict JSON
- Stored fields:
  - `id`
  - `profile`
  - `session_id`
  - `generated_at`
  - `summary`
  - `message_to_user`
  - `message_to_self`
  - `feeling`
  - `focus`
  - `wired`
  - `source`

Definition of done
- Ctrl+B opens a real in-terminal browser in Hermes CLI.
- Browser shows all local profiles as cards, not a one-line list.
- Current profile writes a status file in the background on browser open.
- Status history files accumulate with unique ids.
- Cards update from local state on a short refresh cadence.
- Enter on a card switches sticky active profile and gives clear feedback.
- Structured tests covering browser data/state pass.
- Real smoke validation demonstrates the browser code loads and renders without crashing.

Validation plan
- Targeted pytest for the new browser module and CLI hotkeys.
- Existing nearby CLI/profile tests.
- `python3 -m compileall` on touched Python modules.
- Timed CLI smoke launch under a PTY-sized terminal.
