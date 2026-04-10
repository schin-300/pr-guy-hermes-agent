"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import re
import threading
from typing import Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

_TITLE_SPLIT_RE = re.compile(
    r"\b(?:so that|so|where|when|because|while|but|and then)\b|[\r\n.!?;:]+",
    re.IGNORECASE,
)
_TITLE_WRAPPER_RE = re.compile(r"^(?:\[[^\]]+\]\s*)+")
_TITLE_REQUEST_PREFIX_RE = re.compile(
    r"^(?:(?:can|could|would)\s+you|please|help\s+me(?:\s+with)?|i\s+(?:need|want)\s+you\s+to)\s+",
    re.IGNORECASE,
)
_TITLE_ACTION_PREFIX_RE = re.compile(
    r"^(?:make|do|add|fix|debug|investigate|implement|create|write|update|improve|change|remove|refactor|clean\s+up|rename)\s+"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:simple|small|quick|little|minimal|short)\s+)?"
    r"(?:(?:change|fix|update|improvement|feature|adjustment|cleanup)\s+(?:to|for|around)\s+)?",
    re.IGNORECASE,
)
_TITLE_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_TITLE_TRAILING_SUFFIX_RE = re.compile(
    r"\b(?:functionality|behavior|logic|flow|support|handling|command|feature)\b$",
    re.IGNORECASE,
)

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)


def derive_title_from_text(text: str, max_length: int = 48) -> Optional[str]:
    """Derive a short, readable fallback title directly from user text."""
    cleaned = str(text or "").replace("`", " ").strip()
    if not cleaned:
        return None

    cleaned = _TITLE_WRAPPER_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:;,./")
    if not cleaned:
        return None

    parts = [segment.strip(" -–—:;,./") for segment in _TITLE_SPLIT_RE.split(cleaned)]
    clause = next((segment for segment in parts if segment), cleaned)

    for pattern in (_TITLE_REQUEST_PREFIX_RE, _TITLE_ACTION_PREFIX_RE, _TITLE_LEADING_ARTICLE_RE):
        updated = pattern.sub("", clause).strip(" -–—:;,./")
        if updated:
            clause = updated

    while True:
        updated = _TITLE_TRAILING_SUFFIX_RE.sub("", clause).strip(" -–—:;,./")
        if not updated or updated == clause:
            break
        clause = updated

    clause = re.sub(r"\s+", " ", clause).strip(" -–—:;,./") or cleaned
    title = clause.title().strip()
    if not title:
        return None
    if len(title) <= max_length:
        return title

    truncated = title[: max_length + 1]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip() or title[:max_length].strip()


def generate_title(user_message: str, assistant_response: str, timeout: float = 30.0) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the auxiliary LLM client (cheapest/fastest available model).
    Returns the title string or None on failure.
    """
    # Truncate long messages to keep the request small
    user_snippet = user_message[:500] if user_message else ""
    assistant_snippet = assistant_response[:500] if assistant_response else ""

    messages = [
        {"role": "system", "content": _TITLE_PROMPT},
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]

    fallback_title = derive_title_from_text(user_message)

    try:
        response = call_llm(
            task="compression",  # reuse compression task config (cheap/fast model)
            messages=messages,
            max_tokens=30,
            temperature=0.3,
            timeout=timeout,
        )
        title = (response.choices[0].message.content or "").strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else fallback_title
    except Exception as e:
        logger.debug("Title generation failed: %s", e)
        return fallback_title


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    on_title: Optional[Callable[[str], None]] = None,
) -> None:
    """Generate and set a session title if one doesn't already exist.

    Called in a background thread after the first exchange completes.
    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    """
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title before first response)
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        return

    title = generate_title(user_message, assistant_response)
    if not title:
        return

    try:
        session_db.set_session_title(session_id, title)
        logger.debug("Auto-generated session title: %s", title)
        if on_title:
            try:
                on_title(title)
            except Exception as e:
                logger.debug("Auto-title callback failed: %s", e)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    on_title: Optional[Callable[[str], None]] = None,
) -> None:
    """Fire-and-forget title generation after the first exchange.

    Only generates a title when:
    - This appears to be the first user→assistant exchange
    - No title is already set
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    # Count user messages in history to detect first exchange.
    # conversation_history includes the exchange that just happened,
    # so for a first exchange we expect exactly 1 user message
    # (or 2 counting system). Be generous: generate on first 2 exchanges.
    user_msg_count = sum(1 for m in (conversation_history or []) if m.get("role") == "user")
    if user_msg_count > 2:
        return

    thread_kwargs = {}
    if on_title is not None:
        thread_kwargs["on_title"] = on_title

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response),
        kwargs=thread_kwargs,
        daemon=True,
        name="auto-title",
    )
    thread.start()
