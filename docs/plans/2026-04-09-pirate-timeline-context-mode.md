# Pirate timeline context mode

## Goal
Ship a usable Hermes mode/profile substrate for ultra-long conversations that keeps the full chat as raw chronological transcript data, avoids lossy deep-history dependence, and gives the agent a fast way to scan large recent-or-relevant chunks of the current session lineage.

## User requirements captured
- Big live window: target 1,000,000 token context with compaction threshold 0.95 for this mode.
- Preserve the entire chat with no edits as the source of truth.
- The agent must be able to inspect deep history by date / chronology.
- Retrieval must be fast.
- Retrieval should scan a fat contiguous portion at a time, not tiny isolated snippets.
- More recent chunks should be favored by default.
- It should behave like an immediate context extender.
- The mode should live in a separate cloned profile with separate built-in memory and a separate Honcho AI peer.
- Persona for this mode: lightly pirate-flavored, but not clownish during serious programming.
- Save the workflow as a reusable skill.
- Commit the implementation into the user’s fork so it is usable immediately.

## Approach
1. Add a raw session-lineage retrieval tool for the current conversation family.
   - Search current session + compression continuations by chronological lineage.
   - Support recent-chunk retrieval, date-range browsing, and query-centered chunk retrieval.
   - Return raw unedited transcript entries with timestamps.
   - Favor recent matches by default and expand around a hit into a contiguous chunk.
2. Add a non-lossy compaction mode.
   - Keep existing raw transcript in session lineage unchanged.
   - When compaction triggers in this mode, replace deep middle prompt content with a compact handoff marker that explicitly points the model to the raw timeline tool.
   - Do not summarize/edit older raw transcript for this mode.
3. Add a named context profile for this mode.
   - 1,000,000 context length.
   - 0.95 compaction threshold.
   - Timeline/archive compaction mode.
4. Add profile customization helpers.
   - Allow profile creation to set system prompt and context defaults.
   - Use this to materialize a `pirate-context` profile cloned from default.
   - Separate profile => separate built-in memory files and separate Honcho host/peer block.
5. Create the actual local profile after code lands.
6. Add focused tests.

## Non-goals for this pass
- Full vector embedding infrastructure.
- Replacing all context management in Hermes.
- Automatic learned reranking across semantic/graph/state layers.
- Hosted multi-session retrieval service.

## Likely files
- `agent/context_compressor.py`
- `run_agent.py`
- `hermes_state.py`
- `tools/session_timeline_tool.py` (new)
- `model_tools.py`
- `toolsets.py`
- `hermes_cli/context_limit.py`
- `hermes_cli/profiles.py`
- `hermes_cli/main.py`
- tests under `tests/tools/`, `tests/run_agent/`, `tests/hermes_cli/`, `tests/`

## Definition of done
- Hermes has a raw current-session-lineage retrieval tool that exposes chronological transcript chunks.
- The tool supports recent-first scan, query-centered scan, and date-range lookup.
- A new context mode exists that preserves raw lineage and uses archive-style compaction instead of summary-based middle-turn rewriting.
- A named 1M / 0.95 context profile exists for the mode.
- A cloned local profile with separate Honcho AI peer and pirate-lite system prompt is created.
- Focused tests pass.
- Changes are committed and pushed to the user fork.
- A reusable skill describing how to create/use this mode is saved.

## Validation plan
- Unit tests for timeline helpers and schema.
- Unit tests for lineage search / date ordering / recency bias.
- Unit tests for archive compaction mode marker behavior.
- Unit tests for context mode profile constants.
- Unit tests for profile customization helpers.
- CLI smoke validation creating the profile under a temp/test home.
- Real local profile creation using the patched CLI/module.

## Ordered execution steps
1. Implement session lineage retrieval helpers in `hermes_state.py`.
2. Add new `session_timeline` tool with recent/query/date-range chunk retrieval.
3. Wire the tool into `run_agent.py`, `model_tools.py`, and `toolsets.py`.
4. Add archive compaction mode to `ContextCompressor` + agent wiring.
5. Extend context profiles with a new 1M raw/timeline mode.
6. Add profile customization helpers and actual profile creation flow.
7. Write focused tests.
8. Run focused pytest slices.
9. Create the real `pirate-context` profile locally.
10. Commit, push, save skill, report usage.
