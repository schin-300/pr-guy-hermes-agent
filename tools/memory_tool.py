#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import fcntl
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

# Backward-compatible alias — gateway/run.py imports this at runtime inside
# a function body, so it gets the correct snapshot for that process.  New code
# should prefer get_memory_dir().
MEMORY_DIR = get_memory_dir()

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence via shell rc
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

# Subset of invisible chars for injection detection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}

_COMPACTION_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "your",
    "user", "users", "agent", "hermes", "memory", "profile", "notes", "entry",
    "entries", "them", "they", "their", "there", "what", "when", "where", "which",
    "while", "have", "has", "had", "would", "could", "should", "keep", "uses",
    "old", "note", "notes",
    "using", "used", "just", "more", "most", "very", "still", "will", "before",
    "after", "then", "than", "onto", "over", "under", "same", "only", "also",
    "like", "want", "wants", "need", "needs", "being", "been", "because",
    "avoid", "prefer", "always", "never", "dont", "don't", "doesnt", "doesn't",
}

_COMPACTION_STRONG_SIGNALS = (
    "prefer", "avoid", "don't", "do not", "never", "always", "ask", "must",
    "keep", "use", "first", "approval", "clarify",
)

_COMPACTION_MEDIUM_SIGNALS = (
    "workflow", "terminal", "gui", "repo", "docs", "testing", "review",
    "profile", "memory", "honcho", "branch", "worktree", "ssh", "tmux",
)


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."

    return None


def _normalize_entry_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()



def _tokenize_topic_words(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9][a-z0-9_\-/]*", (text or "").lower())
    tokens = []
    for word in words:
        if len(word) < 3:
            continue
        if word in _COMPACTION_STOPWORDS:
            continue
        tokens.append(word)
    return tokens



def _rank_topic_tokens(text: str, max_words: Optional[int] = None) -> List[str]:
    counts: Dict[str, int] = {}
    ordered_tokens: List[str] = []
    for token in _tokenize_topic_words(text):
        if token not in counts:
            counts[token] = 0
            ordered_tokens.append(token)
        counts[token] += 1

    ranked = sorted(ordered_tokens, key=lambda token: (-counts[token], ordered_tokens.index(token), token))
    if max_words is None:
        return ranked
    return ranked[:max_words]



def _topic_label(text: str, max_words: int = 3) -> str:
    chosen = _rank_topic_tokens(text, max_words=max_words)
    if not chosen:
        fallback = _normalize_entry_text(text)
        return fallback[:48] + ("..." if len(fallback) > 48 else "")
    return ", ".join(chosen)



def _topic_overlap_score(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    union = left_tokens | right_tokens
    jaccard = len(overlap) / max(1, len(union))
    bonus = 0.15 if len(overlap) >= 2 else 0.0
    return jaccard + bonus



def _truncate_words(text: str, limit: int) -> str:
    text = _normalize_entry_text(text)
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    if " " not in text:
        return text[:limit].rstrip(" ,;.")
    clipped = text[: limit + 1]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;.")



def _split_clauses(text: str) -> List[str]:
    normalized = _normalize_entry_text(text)
    if not normalized:
        return []
    parts = re.split(r";|\.(?=\s+[A-Z`/])", normalized)
    clauses = []
    for part in parts:
        clause = part.strip(" .;")
        if clause:
            clauses.append(clause)
    return clauses or [normalized]



def _clause_score(clause: str) -> float:
    lower = clause.lower()
    score = 0.0
    if any(signal in lower for signal in _COMPACTION_STRONG_SIGNALS):
        score += 4.0
    if any(signal in lower for signal in _COMPACTION_MEDIUM_SIGNALS):
        score += 2.0
    if any(ch in clause for ch in ("`", "/")):
        score += 1.0
    score -= len(clause) / 180.0
    return score



def _compact_entry_text(target: str, text: str) -> str:
    normalized = _normalize_entry_text(text)
    if not normalized:
        return normalized

    target_len = int(len(normalized) * (0.68 if target == "user" else 0.72))
    minimum_len = 58 if target == "user" else 80
    target_len = max(minimum_len, min(len(normalized) - 16, target_len))
    if target_len >= len(normalized):
        return normalized

    clauses = _split_clauses(normalized)
    if len(clauses) == 1:
        return _truncate_words(normalized, target_len)

    chosen: List[tuple[int, str]] = [(0, clauses[0])]
    remaining = list(enumerate(clauses[1:], start=1))
    remaining.sort(key=lambda item: (-_clause_score(item[1]), len(item[1]), item[0]))

    for idx, clause in remaining:
        candidate = "; ".join(part for _, part in sorted(chosen + [(idx, clause)]))
        if len(candidate) <= target_len:
            chosen.append((idx, clause))

    compacted = "; ".join(part for _, part in sorted(chosen))
    compacted = _truncate_words(compacted, target_len)
    return compacted if len(compacted) < len(normalized) else normalized



def _entry_value_score(target: str, entry: str) -> float:
    lower = entry.lower()
    score = 0.0
    if any(signal in lower for signal in _COMPACTION_STRONG_SIGNALS):
        score += 4.0
    if any(signal in lower for signal in _COMPACTION_MEDIUM_SIGNALS):
        score += 2.0
    if any(ch in entry for ch in ("`", "/")):
        score += 1.2
    if target == "memory" and re.search(r"/|\.json|\.yaml|\.yml|\.md|localhost|127\.0\.0\.1", lower):
        score -= 1.0
    score -= len(entry) / (170.0 if target == "user" else 220.0)
    return score


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot."""
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(lock_path, "w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        """
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def _projected_total_after_write(
        self,
        target: str,
        incoming_content: str,
        action: str = "add",
        old_text: Optional[str] = None,
        entries: Optional[List[str]] = None,
    ) -> int:
        entries = list(entries if entries is not None else self._entries_for(target))
        incoming = _normalize_entry_text(incoming_content)
        if action == "replace" and old_text:
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
            if len(matches) == 1:
                updated = entries.copy()
                updated[matches[0][0]] = incoming
                return len(ENTRY_DELIMITER.join(updated)) if updated else 0
        updated = entries + ([incoming] if incoming else [])
        return len(ENTRY_DELIMITER.join(updated)) if updated else 0

    @staticmethod
    def _entries_char_count(entries: List[str]) -> int:
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def _chars_saved_by_removing_entries(self, target: str, entries_to_remove: List[str]) -> int:
        current_entries = list(self._entries_for(target))
        current_total = self._entries_char_count(current_entries)
        remaining = list(current_entries)
        for old_entry in entries_to_remove:
            normalized = _normalize_entry_text(old_entry)
            idx = next((i for i, entry in enumerate(remaining) if _normalize_entry_text(entry) == normalized), None)
            if idx is None:
                idx = next((i for i, entry in enumerate(remaining) if normalized and normalized in _normalize_entry_text(entry)), None)
            if idx is not None:
                remaining.pop(idx)
        return max(0, current_total - self._entries_char_count(remaining))

    def _build_topic_groups(self, target: str, incoming_content: str, target_chars_to_free: int) -> List[Dict[str, Any]]:
        entries = list(self._entries_for(target))
        incoming_tokens = set(_rank_topic_tokens(incoming_content, max_words=8))
        groups: List[Dict[str, Any]] = []

        for entry in entries:
            tokens = set(_rank_topic_tokens(entry, max_words=8))
            best_idx = None
            best_score = 0.0
            for idx, group in enumerate(groups):
                score = _topic_overlap_score(tokens, group["tokens"])
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score >= 0.25:
                group = groups[best_idx]
                group["entries"].append(entry)
                group["tokens"].update(tokens)
            else:
                groups.append({
                    "entries": [entry],
                    "tokens": set(tokens),
                })

        ranked_groups: List[Dict[str, Any]] = []
        for group in groups:
            group_entries = group["entries"]
            combined_text = " ".join(group_entries)
            ranked_tokens = _rank_topic_tokens(combined_text, max_words=3)
            topic = ", ".join(ranked_tokens) if ranked_tokens else _topic_label(combined_text)
            chars_freed = self._chars_saved_by_removing_entries(target, group_entries)
            if chars_freed <= 0:
                continue
            avg_value = sum(_entry_value_score(target, entry) for entry in group_entries) / max(1, len(group_entries))
            overlap_tokens = [token for token in ranked_tokens if token in incoming_tokens]
            if overlap_tokens:
                reason = f"Topic overlaps with the incoming memory on {', '.join(overlap_tokens[:2])}; delete only if this whole topic is stale."
            else:
                reason = "Potentially stale topic cluster with no strong overlap to the new memory."
            ranked_groups.append({
                "topic": topic,
                "entry_count": len(group_entries),
                "chars_freed_if_removed": chars_freed,
                "entries": list(group_entries),
                "reason": reason,
                "overlap_tokens": overlap_tokens,
                "_sort": (1 if overlap_tokens else 0, avg_value, -chars_freed),
            })

        ranked_groups.sort(key=lambda group: group.get("_sort", (99, 99, 99)))

        selected_groups: List[Dict[str, Any]] = []
        cumulative_freed = 0
        for idx, group in enumerate(ranked_groups, start=1):
            selected_groups.append({
                "id": idx,
                "topic": group["topic"],
                "entry_count": group["entry_count"],
                "chars_freed_if_removed": group["chars_freed_if_removed"],
                "entries": group["entries"],
                "reason": group["reason"],
                "overlap_tokens": group["overlap_tokens"],
            })
            cumulative_freed += int(group["chars_freed_if_removed"])
            if len(selected_groups) >= 6:
                break
            if cumulative_freed >= max(target_chars_to_free, 1) and len(selected_groups) >= 2:
                break

        return selected_groups

    def build_topic_group_selection_proposal(
        self,
        target: str,
        proposal: Dict[str, Any],
        selected_group_ids: List[int],
    ) -> Dict[str, Any]:
        topic_groups = proposal.get("topic_groups", []) if isinstance(proposal, dict) else []
        group_map = {
            int(group.get("id")): group
            for group in topic_groups
            if str(group.get("id", "")).isdigit()
        }
        matched_groups: List[Dict[str, Any]] = []
        for group_id in selected_group_ids:
            group = group_map.get(int(group_id))
            if group and group not in matched_groups:
                matched_groups.append(group)

        if not matched_groups:
            return {"success": False, "error": "No valid topic groups were selected."}

        selected_entries: List[str] = []
        seen_entries = set()
        candidates: List[Dict[str, Any]] = []
        for group in matched_groups:
            topic = str(group.get("topic", "memory") or "memory")
            group_id = int(group.get("id"))
            for entry in group.get("entries", []) or []:
                normalized = _normalize_entry_text(entry)
                if not normalized or normalized in seen_entries:
                    continue
                seen_entries.add(normalized)
                selected_entries.append(entry)
                candidates.append({
                    "kind": "remove",
                    "topic": topic,
                    "group_id": group_id,
                    "old_entry": entry,
                    "new_entry": None,
                    "chars_saved": len(entry) + len(ENTRY_DELIMITER),
                    "reason": f"Removed by user-selected topic group #{group_id}.",
                })

        freed_chars = self._chars_saved_by_removing_entries(target, selected_entries)
        current_usage = int(proposal.get("current_usage", self._char_count(target)) or self._char_count(target))
        projected_after_write = int(proposal.get("projected_after_write", current_usage) or current_usage)
        limit = int(proposal.get("limit", self._char_limit(target)) or self._char_limit(target))
        estimated_usage_after_compaction = max(current_usage - freed_chars, 0)
        estimated_usage_after_retry = max(projected_after_write - freed_chars, 0)

        return {
            "success": True,
            "target": target,
            "limit": limit,
            "current_usage": current_usage,
            "projected_after_write": projected_after_write,
            "required_chars": max(0, projected_after_write - limit),
            "target_chars_to_free": int(proposal.get("target_chars_to_free", 0) or 0),
            "freed_chars": freed_chars,
            "estimated_usage_after_compaction": estimated_usage_after_compaction,
            "estimated_usage_after_retry": estimated_usage_after_retry,
            "can_make_room": estimated_usage_after_retry <= limit,
            "candidates": candidates,
            "topic_groups": topic_groups,
            "selected_group_ids": [int(group.get("id")) for group in matched_groups],
            "incoming_content": proposal.get("incoming_content", ""),
            "action": proposal.get("action", "add"),
            "old_text": proposal.get("old_text"),
        }

    def build_compaction_proposal(
        self,
        target: str,
        incoming_content: str,
        action: str = "add",
        old_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        entries = list(self._entries_for(target))
        limit = self._char_limit(target)
        current = self._char_count(target)
        projected_after_write = self._projected_total_after_write(
            target,
            incoming_content,
            action=action,
            old_text=old_text,
            entries=entries,
        )
        required_chars = max(0, projected_after_write - limit)
        desired_headroom = max(24, min(120, int(limit * 0.08)))
        target_chars_to_free = required_chars + desired_headroom if required_chars > 0 else desired_headroom

        candidate_pool: List[Dict[str, Any]] = []
        seen_normalized: Dict[str, str] = {}
        touched_entries = set()

        for entry in entries:
            normalized = _normalize_entry_text(entry).lower()
            if normalized in seen_normalized:
                candidate_pool.append({
                    "kind": "remove",
                    "topic": _topic_label(entry),
                    "old_entry": entry,
                    "new_entry": None,
                    "chars_saved": len(entry) + len(ENTRY_DELIMITER),
                    "reason": "Exact duplicate entry.",
                    "_sort": (0, _entry_value_score(target, entry), -(len(entry) + len(ENTRY_DELIMITER))),
                })
            else:
                seen_normalized[normalized] = entry

        for entry in entries:
            compacted = _compact_entry_text(target, entry)
            if not compacted or compacted == entry:
                continue
            saved = len(entry) - len(compacted)
            if saved < 16:
                continue
            candidate_pool.append({
                "kind": "shorten",
                "topic": _topic_label(entry),
                "old_entry": entry,
                "new_entry": compacted,
                "chars_saved": saved,
                "reason": "Verbose entry can be shortened while keeping the core instruction.",
                "_sort": (1, _entry_value_score(target, entry), -saved),
            })

        allow_user_removals = target != "user"
        shorten_capacity = sum(c["chars_saved"] for c in candidate_pool if c["kind"] == "shorten")
        if required_chars > max(0, shorten_capacity):
            allow_user_removals = True

        if allow_user_removals:
            for entry in entries:
                candidate_pool.append({
                    "kind": "remove",
                    "topic": _topic_label(entry),
                    "old_entry": entry,
                    "new_entry": None,
                    "chars_saved": len(entry) + len(ENTRY_DELIMITER),
                    "reason": "Lowest-priority entry if more room is needed.",
                    "_sort": (2, _entry_value_score(target, entry), -(len(entry) + len(ENTRY_DELIMITER))),
                })

        candidate_pool.sort(key=lambda candidate: candidate.get("_sort", (99, 99, 99)))

        selected: List[Dict[str, Any]] = []
        freed_chars = 0
        for candidate in candidate_pool:
            old_entry = candidate.get("old_entry", "")
            if old_entry in touched_entries:
                continue
            selected.append({k: v for k, v in candidate.items() if not k.startswith("_")})
            touched_entries.add(old_entry)
            freed_chars += int(candidate.get("chars_saved", 0))
            if freed_chars >= target_chars_to_free:
                break

        estimated_usage_after_compaction = max(current - freed_chars, 0)
        estimated_usage_after_retry = max(projected_after_write - freed_chars, 0)
        topic_groups = self._build_topic_groups(target, incoming_content, target_chars_to_free)
        return {
            "success": True,
            "target": target,
            "limit": limit,
            "current_usage": current,
            "projected_after_write": projected_after_write,
            "required_chars": required_chars,
            "target_chars_to_free": target_chars_to_free,
            "freed_chars": freed_chars,
            "estimated_usage_after_compaction": estimated_usage_after_compaction,
            "estimated_usage_after_retry": estimated_usage_after_retry,
            "can_make_room": estimated_usage_after_retry <= limit,
            "candidates": selected,
            "topic_groups": topic_groups,
            "incoming_content": _normalize_entry_text(incoming_content),
            "action": action,
            "old_text": old_text,
        }

    def apply_compaction_proposal(self, target: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
        candidates = proposal.get("candidates") if isinstance(proposal, dict) else None
        if not candidates:
            return {"success": False, "error": "Compaction proposal is empty."}

        applied = []
        with self._file_lock(self._path_for(target)):
            self._reload_target(target)
            entries = list(self._entries_for(target))

            for candidate in candidates:
                old_entry = _normalize_entry_text(candidate.get("old_entry", ""))
                if not old_entry:
                    continue
                idx = next((i for i, entry in enumerate(entries) if _normalize_entry_text(entry) == old_entry), None)
                if idx is None:
                    idx = next((i for i, entry in enumerate(entries) if old_entry in entry), None)
                if idx is None:
                    continue

                kind = candidate.get("kind")
                if kind == "shorten":
                    new_entry = _normalize_entry_text(candidate.get("new_entry", ""))
                    if not new_entry:
                        continue
                    entries[idx] = new_entry
                    applied.append({"kind": kind, "topic": candidate.get("topic", ""), "old_entry": old_entry, "new_entry": new_entry})
                elif kind == "remove":
                    entries.pop(idx)
                    applied.append({"kind": kind, "topic": candidate.get("topic", ""), "old_entry": old_entry})

            entries = [_normalize_entry_text(entry) for entry in entries if _normalize_entry_text(entry)]
            entries = list(dict.fromkeys(entries))
            self._set_entries(target, entries)
            self.save_to_disk(target)

        resp = self._success_response(target, "Compaction applied.")
        resp["compaction_applied"] = applied
        return resp

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions
            self._reload_target(target)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))  # Atomic on same filesystem
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




