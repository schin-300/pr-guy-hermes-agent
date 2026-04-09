"""Public Hermes docs/code lookup tool.

Searches the public upstream Hermes repository (NousResearch/hermes-agent@main)
with docs-first ranking so the agent can answer Hermes product questions without
being biased toward the current local checkout/fork.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List

import requests

from hermes_constants import get_hermes_home
from tools.registry import registry
from utils import atomic_json_write

logger = logging.getLogger(__name__)

PUBLIC_REPO = "NousResearch/hermes-agent"
PUBLIC_BRANCH = "main"
PUBLIC_TREE_URL = f"https://api.github.com/repos/{PUBLIC_REPO}/git/trees/{PUBLIC_BRANCH}?recursive=1"
PUBLIC_RAW_BASE_URL = f"https://raw.githubusercontent.com/{PUBLIC_REPO}/{PUBLIC_BRANCH}/"
PUBLIC_BLOB_BASE_URL = f"https://github.com/{PUBLIC_REPO}/blob/{PUBLIC_BRANCH}/"
PUBLIC_DOCS_BASE_URL = "https://hermes-agent.nousresearch.com/docs/"

TREE_TTL_SECONDS = 60 * 60
CONTENT_TTL_SECONDS = 60 * 60 * 24
DEFAULT_LIMIT = 5
MAX_LIMIT = 10
MAX_CANDIDATE_FILES = 18
MIN_CANDIDATE_FILES = 8

VALID_SCOPES = {"docs", "code", "docs_and_code"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "config", "configuration",
    "difference", "do", "does", "for", "from", "get", "how", "i", "in", "is", "it", "kind", "kinds",
    "of", "on", "or", "our", "should", "that", "the", "their", "them", "these", "this", "to", "up",
    "use", "using", "what", "when", "which", "why", "with", "would", "you", "your",
}

TOKEN_SYNONYMS = {
    "plugin": {"plugins"},
    "plugins": {"plugin"},
    "skin": {"skins", "theme", "themes"},
    "skins": {"skin", "theme", "themes"},
    "theme": {"themes", "skin", "skins"},
    "themes": {"theme", "skin", "skins"},
    "personality": {"soul"},
    "memory": {"memories"},
    "provider": {"providers"},
    "providers": {"provider"},
    "tool": {"tools", "toolset", "toolsets"},
    "tools": {"tool", "toolset", "toolsets"},
    "toolset": {"toolsets", "tools"},
    "toolsets": {"toolset", "tools"},
    "command": {"commands", "cli"},
    "commands": {"command", "cli"},
    "docs": {"documentation"},
    "documentation": {"docs"},
    "lookup": {"search"},
    "search": {"lookup"},
}

CURATED_FALLBACK_PATHS = [
    "website/docs/getting-started/quickstart.md",
    "website/docs/user-guide/configuration.md",
    "website/docs/user-guide/features/overview.md",
    "website/docs/reference/cli-commands.md",
    "website/docs/reference/slash-commands.md",
    "website/docs/reference/tools-reference.md",
    "website/docs/developer-guide/architecture.md",
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "run_agent.py",
    "cli.py",
    "model_tools.py",
    "toolsets.py",
]

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "HermesAgent/hermes_docs_lookup"})

_TREE_MEMORY_CACHE: dict[str, Any] = {"tree": None, "fetched_at": 0.0}
_CONTENT_MEMORY_CACHE: dict[str, tuple[float, str]] = {}

_ROOT_DOCS = {"README.md", "AGENTS.md", "CONTRIBUTING.md"}
_CODE_ROOTS = (
    "agent/",
    "gateway/",
    "hermes_cli/",
    "tools/",
)
_CODE_ROOT_FILES = {"run_agent.py", "cli.py", "model_tools.py", "toolsets.py"}


def _cache_dir() -> Path:
    path = get_hermes_home() / "cache" / "hermes_docs_lookup"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tree_cache_path() -> Path:
    return _cache_dir() / "public_tree.json"


def _content_cache_path(path: str) -> Path:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
    return _cache_dir() / "content" / f"{digest}.json"


def _load_cache_payload(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to load cache payload %s: %s", path, exc)
    return None


def _save_cache_payload(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, payload, indent=None, separators=(",", ":"))
    except Exception as exc:
        logger.debug("Failed to save cache payload %s: %s", path, exc)


def _cache_is_fresh(payload: dict[str, Any] | None, ttl_seconds: int) -> bool:
    if not isinstance(payload, dict):
        return False
    fetched_at = payload.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return False
    return (time.time() - float(fetched_at)) < ttl_seconds


def _http_get_json(url: str, timeout: int = 20) -> Any:
    response = _SESSION.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _http_get_text(url: str, timeout: int = 20) -> str:
    response = _SESSION.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def _fetch_upstream_tree(*, prefer_cache: bool = True) -> list[dict[str, Any]]:
    if prefer_cache and _TREE_MEMORY_CACHE.get("tree") and (time.time() - _TREE_MEMORY_CACHE.get("fetched_at", 0.0)) < TREE_TTL_SECONDS:
        return list(_TREE_MEMORY_CACHE["tree"])

    disk_payload = _load_cache_payload(_tree_cache_path()) if prefer_cache else None
    if prefer_cache and _cache_is_fresh(disk_payload, TREE_TTL_SECONDS):
        tree = disk_payload.get("tree") or []
        _TREE_MEMORY_CACHE["tree"] = list(tree)
        _TREE_MEMORY_CACHE["fetched_at"] = float(disk_payload.get("fetched_at", time.time()))
        return list(tree)

    try:
        data = _http_get_json(PUBLIC_TREE_URL)
        tree = data.get("tree") or []
        payload = {"fetched_at": time.time(), "tree": tree}
        _TREE_MEMORY_CACHE["tree"] = list(tree)
        _TREE_MEMORY_CACHE["fetched_at"] = float(payload["fetched_at"])
        _save_cache_payload(_tree_cache_path(), payload)
        return list(tree)
    except Exception as exc:
        logger.debug("Failed to fetch public Hermes tree: %s", exc)

    if isinstance(disk_payload, dict):
        tree = disk_payload.get("tree") or []
        if tree:
            return list(tree)
    return []


def _fetch_upstream_text(path: str, *, prefer_cache: bool = True) -> str:
    if prefer_cache:
        mem_entry = _CONTENT_MEMORY_CACHE.get(path)
        if mem_entry and (time.time() - mem_entry[0]) < CONTENT_TTL_SECONDS:
            return mem_entry[1]

    cache_path = _content_cache_path(path)
    disk_payload = _load_cache_payload(cache_path) if prefer_cache else None
    if prefer_cache and _cache_is_fresh(disk_payload, CONTENT_TTL_SECONDS):
        text = str(disk_payload.get("text") or "")
        _CONTENT_MEMORY_CACHE[path] = (float(disk_payload.get("fetched_at", time.time())), text)
        return text

    url = f"{PUBLIC_RAW_BASE_URL}{path}"
    try:
        text = _http_get_text(url)
        payload = {"fetched_at": time.time(), "path": path, "text": text}
        _CONTENT_MEMORY_CACHE[path] = (float(payload["fetched_at"]), text)
        _save_cache_payload(cache_path, payload)
        return text
    except Exception as exc:
        logger.debug("Failed to fetch upstream text for %s: %s", path, exc)

    if isinstance(disk_payload, dict) and isinstance(disk_payload.get("text"), str):
        return str(disk_payload["text"])
    raise


def _local_checkout_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_local_paths(scope: str) -> list[dict[str, Any]]:
    root = _local_checkout_root()
    paths: list[dict[str, Any]] = []

    def _add_if_exists(rel: str) -> None:
        if (root / rel).exists():
            paths.append({"path": rel, "type": "blob"})

    if scope in {"docs", "docs_and_code"}:
        docs_root = root / "website" / "docs"
        if docs_root.exists():
            for file_path in docs_root.rglob("*.md"):
                paths.append({"path": str(file_path.relative_to(root)), "type": "blob"})
            for file_path in docs_root.rglob("*.mdx"):
                paths.append({"path": str(file_path.relative_to(root)), "type": "blob"})
        for rel in sorted(_ROOT_DOCS):
            _add_if_exists(rel)
        for file_path in sorted(root.glob("RELEASE_*.md")):
            paths.append({"path": str(file_path.relative_to(root)), "type": "blob"})

    if scope in {"code", "docs_and_code"}:
        for rel in sorted(_CODE_ROOT_FILES):
            _add_if_exists(rel)
        for subdir in _CODE_ROOTS:
            base = root / subdir
            if base.exists():
                for file_path in base.rglob("*.py"):
                    paths.append({"path": str(file_path.relative_to(root)), "type": "blob"})

    return paths


def _fetch_local_text(path: str, *, prefer_cache: bool = True) -> str:
    del prefer_cache
    return (_local_checkout_root() / path).read_text(encoding="utf-8")


def _is_docs_path(path: str) -> bool:
    return path.startswith("website/docs/") and path.endswith((".md", ".mdx"))


def _is_root_doc_path(path: str) -> bool:
    return path in _ROOT_DOCS or bool(re.fullmatch(r"RELEASE_[^/]+\.md", path))


def _is_code_path(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    return path in _CODE_ROOT_FILES or any(path.startswith(prefix) for prefix in _CODE_ROOTS)


def _path_allowed_for_scope(path: str, scope: str) -> bool:
    if scope == "docs":
        return _is_docs_path(path) or _is_root_doc_path(path)
    if scope == "code":
        return _is_code_path(path)
    if scope == "docs_and_code":
        return _is_docs_path(path) or _is_root_doc_path(path) or _is_code_path(path)
    return False


def _source_kind_for_path(path: str, corpus: str) -> str:
    if corpus == "public_hermes_main":
        if _is_docs_path(path):
            return "public_docs"
        if _is_root_doc_path(path):
            return "public_root_docs"
        return "public_code"

    if _is_docs_path(path):
        return "local_checkout_fallback_docs"
    if _is_root_doc_path(path):
        return "local_checkout_fallback_root_docs"
    return "local_checkout_fallback_code"


def _normalize_token(token: str) -> set[str]:
    token = token.strip().lower()
    if not token:
        return set()

    variants = {token}
    if len(token) > 4 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    if len(token) > 3 and token.endswith("es"):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.add(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        variants.add(token[:-3])
    if len(token) > 4 and token.endswith("ed"):
        variants.add(token[:-2])
    variants |= TOKEN_SYNONYMS.get(token, set())
    return {v for v in variants if v and v not in STOPWORDS}


def _tokenize(text: str) -> set[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    tokens: set[str] = set()
    for raw in raw_tokens:
        tokens |= _normalize_token(raw)
        if "_" in raw:
            for part in raw.split("_"):
                tokens |= _normalize_token(part)
    return tokens


def _query_terms(query: str) -> set[str]:
    tokens = _tokenize(query)
    if not tokens:
        stripped = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        return {tok for tok in stripped if tok}
    return tokens


def _docs_page_url(path: str, heading: str | None = None) -> str:
    rel = path.removeprefix("website/docs/")
    rel = re.sub(r"\.(md|mdx)$", "", rel)
    url = f"{PUBLIC_DOCS_BASE_URL}{rel}"
    if heading:
        anchor = _slugify_heading(heading)
        if anchor:
            url = f"{url}#{anchor}"
    return url


def _github_blob_url(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    url = f"{PUBLIC_BLOB_BASE_URL}{path}"
    if start_line and end_line and end_line >= start_line:
        if end_line == start_line:
            return f"{url}#L{start_line}"
        return f"{url}#L{start_line}-L{end_line}"
    if start_line:
        return f"{url}#L{start_line}"
    return url


def _slugify_heading(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[`*_]+", "", slug)
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _path_priority(path: str) -> int:
    if _is_docs_path(path):
        return 0
    if _is_root_doc_path(path):
        return 1
    return 2


def _looks_like_symbol_query(query: str) -> bool:
    return bool(re.search(r"[_./:]", query)) or bool(re.search(r"[a-z][A-Z]", query))


def _path_score(path: str, query_terms: set[str], query: str) -> float:
    path_tokens = _tokenize(path.replace("/", " ").replace("-", " ").replace(".", " "))
    filename = Path(path).stem.lower()
    score = 0.0
    matched_terms = 0
    path_lower = path.lower()
    symbol_query = _looks_like_symbol_query(query)
    for term in query_terms:
        if term in path_tokens:
            score += 4.0
            matched_terms += 1
        elif term in path_lower:
            score += 1.5
        if term == filename:
            score += 5.0
    score += float(matched_terms * matched_terms)

    if _is_docs_path(path):
        score += 3.0
        if "/user-guide/" in path_lower or "/reference/" in path_lower:
            score += 1.5
        if "/developer-guide/" in path_lower and not symbol_query and not ({"developer", "implementation", "internal", "internals", "runtime", "source", "code"} & query_terms):
            score -= 2.5
    elif _is_root_doc_path(path):
        score += 1.5
    else:
        score += 0.5
        if symbol_query and (query.lower() in path_lower or filename == query.lower()):
            score += 8.0
    return score


@dataclass
class Chunk:
    path: str
    source_kind: str
    title: str
    heading: str
    start_line: int
    end_line: int
    text: str
    url: str
    path_score: float


def _strip_frontmatter(lines: list[str]) -> tuple[list[str], int, str | None]:
    if not lines or lines[0].strip() != "---":
        return lines, 0, None

    title: str | None = None
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == "---":
            body = lines[idx + 1 :]
            for meta_line in lines[1:idx]:
                match = re.match(r"title\s*:\s*[\"']?(.*?)[\"']?\s*$", meta_line.strip())
                if match and match.group(1).strip():
                    title = match.group(1).strip()
                    break
            return body, idx + 1, title
    return lines, 0, None


def _markdown_chunks(path: str, source_kind: str, raw_text: str, path_score: float) -> list[Chunk]:
    lines = raw_text.splitlines()
    body_lines, offset, frontmatter_title = _strip_frontmatter(lines)
    if not body_lines:
        return []

    page_title = frontmatter_title or Path(path).stem.replace("-", " ").title()
    chunks: list[Chunk] = []
    current_heading = page_title
    current_start = 1 + offset
    current_buffer: list[str] = []

    def _flush(end_line_exclusive: int) -> None:
        if not current_buffer:
            return
        text = "\n".join(current_buffer).strip()
        if not text:
            return
        url = _docs_page_url(path, None if current_heading == page_title else current_heading)
        chunks.append(
            Chunk(
                path=path,
                source_kind=source_kind,
                title=page_title,
                heading=current_heading,
                start_line=current_start,
                end_line=max(current_start, end_line_exclusive - 1),
                text=text,
                url=url,
                path_score=path_score,
            )
        )

    for idx, line in enumerate(body_lines, start=1 + offset):
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if heading_match:
            _flush(idx)
            current_heading = heading_match.group(2).strip() or page_title
            current_start = idx
            current_buffer = [line]
        else:
            current_buffer.append(line)

    _flush(len(body_lines) + offset + 1)
    return chunks


def _code_chunks(path: str, source_kind: str, raw_text: str, path_score: float) -> list[Chunk]:
    lines = raw_text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    title = Path(path).name
    symbol_pattern = re.compile(r"^(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)")

    # Module header chunk
    header_end = min(len(lines), 40)
    header_text = "\n".join(lines[:header_end]).strip()
    if header_text:
        chunks.append(
            Chunk(
                path=path,
                source_kind=source_kind,
                title=title,
                heading=f"module {title}",
                start_line=1,
                end_line=header_end,
                text=header_text,
                url=_github_blob_url(path, 1, header_end),
                path_score=path_score,
            )
        )

    symbol_lines: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        match = symbol_pattern.match(line)
        if match:
            symbol_lines.append((idx, match.group(2)))

    for pos, (start_line, symbol_name) in enumerate(symbol_lines):
        next_start = symbol_lines[pos + 1][0] if pos + 1 < len(symbol_lines) else len(lines) + 1
        end_line = min(len(lines), next_start - 1)
        text = "\n".join(lines[start_line - 1 : end_line]).strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                path=path,
                source_kind=source_kind,
                title=title,
                heading=symbol_name,
                start_line=start_line,
                end_line=end_line,
                text=text,
                url=_github_blob_url(path, start_line, end_line),
                path_score=path_score,
            )
        )

    return chunks


def _chunks_for_document(path: str, source_kind: str, text: str, path_score: float) -> list[Chunk]:
    if path.endswith((".md", ".mdx")):
        return _markdown_chunks(path, source_kind, text, path_score)
    return _code_chunks(path, source_kind, text, path_score)


def _combined_text_tokens(chunk: Chunk) -> set[str]:
    return _tokenize(f"{chunk.path} {chunk.title} {chunk.heading} {chunk.text}")


def _chunk_score(chunk: Chunk, query_terms: set[str], query: str) -> float:
    title_tokens = _tokenize(chunk.title)
    heading_tokens = _tokenize(chunk.heading)
    body_tokens = _tokenize(chunk.text)
    combined_tokens = title_tokens | heading_tokens | body_tokens
    symbol_query = _looks_like_symbol_query(query)

    score = chunk.path_score
    for term in query_terms:
        if term in title_tokens:
            score += 7.0
        if term in heading_tokens:
            score += 9.0
        if term in body_tokens:
            score += 2.0
        if term in combined_tokens:
            score += 0.5

    query_words = [w for w in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if w not in STOPWORDS]
    phrase = " ".join(query_words).strip()
    haystack = re.sub(r"\s+", " ", f"{chunk.heading} {chunk.text}".lower())
    if phrase and phrase in haystack:
        score += 10.0

    if symbol_query:
        if query.lower() in chunk.heading.lower():
            score += 12.0
        if query.lower() in chunk.path.lower():
            score += 10.0
        if chunk.source_kind.endswith("code"):
            score += 4.0

    if chunk.source_kind == "public_docs":
        score += 4.0
    elif chunk.source_kind == "public_root_docs":
        score += 2.0

    return score


def _snippet_from_chunk(chunk: Chunk, query_terms: set[str], max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", chunk.text).strip()
    if len(text) <= max_chars:
        return text

    lower = text.lower()
    best_index = -1
    for term in sorted(query_terms, key=len, reverse=True):
        best_index = lower.find(term.lower())
        if best_index != -1:
            break

    if best_index == -1:
        return text[: max_chars - 1].rstrip() + "…"

    start = max(0, best_index - max_chars // 3)
    end = min(len(text), start + max_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet.rstrip() + "…"
    return snippet


def _fallback_candidates(tree: list[dict[str, Any]], scope: str) -> list[str]:
    allowed = {entry.get("path") for entry in tree if isinstance(entry, dict) and _path_allowed_for_scope(str(entry.get("path", "")), scope)}
    return [path for path in CURATED_FALLBACK_PATHS if path in allowed and _path_allowed_for_scope(path, scope)]


def _candidate_paths(tree: list[dict[str, Any]], query_terms: set[str], scope: str, query: str) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = str(entry.get("path") or "")
        if not path or not _path_allowed_for_scope(path, scope):
            continue
        score = _path_score(path, query_terms, query)
        scored.append((path, score))

    scored.sort(key=lambda item: (-item[1], _path_priority(item[0]), item[0]))
    nonzero = [item for item in scored if item[1] > 0][:MAX_CANDIDATE_FILES]
    if len(nonzero) >= MIN_CANDIDATE_FILES:
        return nonzero

    seen = {path for path, _ in nonzero}
    results = list(nonzero)
    for path in _fallback_candidates(tree, scope):
        if path not in seen:
            results.append((path, _path_score(path, query_terms, query)))
            seen.add(path)
        if len(results) >= MIN_CANDIDATE_FILES:
            break

    if len(results) < MIN_CANDIDATE_FILES:
        for path, score in scored:
            if path in seen:
                continue
            results.append((path, score))
            seen.add(path)
            if len(results) >= MIN_CANDIDATE_FILES:
                break

    return results[:MAX_CANDIDATE_FILES]


def _search_tree(
    query: str,
    scope: str,
    limit: int,
    tree: list[dict[str, Any]],
    fetch_text: Callable[..., str],
    corpus: str,
) -> list[dict[str, Any]]:
    query_terms = _query_terms(query)
    candidates = _candidate_paths(tree, query_terms, scope, query)
    scored_chunks: list[tuple[float, Chunk]] = []

    for path, path_score in candidates:
        try:
            text = fetch_text(path, prefer_cache=True)
        except Exception as exc:
            logger.debug("Skipping candidate %s: %s", path, exc)
            continue

        source_kind = _source_kind_for_path(path, corpus)
        for chunk in _chunks_for_document(path, source_kind, text, path_score):
            score = _chunk_score(chunk, query_terms, query)
            if score <= 0:
                continue
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: (-item[0], _path_priority(item[1].path), item[1].path, item[1].start_line))

    results: list[dict[str, Any]] = []
    seen_sections: set[tuple[str, str, int]] = set()
    for score, chunk in scored_chunks:
        section_key = (chunk.path, chunk.heading, chunk.start_line)
        if section_key in seen_sections:
            continue
        seen_sections.add(section_key)
        results.append(
            {
                "path": chunk.path,
                "source_kind": chunk.source_kind,
                "title": chunk.title,
                "heading": chunk.heading,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": round(score, 2),
                "snippet": _snippet_from_chunk(chunk, query_terms),
                "url": chunk.url,
            }
        )
        if len(results) >= limit:
            break

    return results


def hermes_docs_lookup_tool(query: str, scope: str = "docs", limit: int = DEFAULT_LIMIT) -> str:
    """Search public Hermes upstream docs/code and return cited results as JSON."""
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    scope = (scope or "docs").strip().lower()
    if scope not in VALID_SCOPES:
        return json.dumps({"error": f"Invalid scope '{scope}'. Expected one of: {', '.join(sorted(VALID_SCOPES))}"}, ensure_ascii=False)

    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        limit_int = DEFAULT_LIMIT
    limit_int = max(1, min(MAX_LIMIT, limit_int))

    corpus = "public_hermes_main"
    tree = _fetch_upstream_tree(prefer_cache=True)
    fetch_text = _fetch_upstream_text

    if not tree:
        corpus = "local_checkout_fallback"
        tree = _iter_local_paths(scope)
        fetch_text = _fetch_local_text

    results = _search_tree(query, scope, limit_int, tree, fetch_text, corpus)
    payload: dict[str, Any] = {
        "success": True,
        "query": query,
        "scope": scope,
        "corpus": corpus,
        "repo": f"{PUBLIC_REPO}@{PUBLIC_BRANCH}",
        "results": results,
    }
    if corpus != "public_hermes_main":
        payload["warning"] = "Public upstream docs/code were unavailable; results are from the local checkout fallback and may differ from public Hermes."

    return json.dumps(payload, ensure_ascii=False)


HERMES_DOCS_LOOKUP_SCHEMA = {
    "name": "hermes_docs_lookup",
    "description": (
        "Search public Hermes upstream documentation and selected upstream source files from "
        "NousResearch/hermes-agent main. Use this first for Hermes Agent questions about "
        "commands, config, features, docs, architecture, or upstream code behavior instead of "
        "searching the local checkout. Returns cited sections with URLs and source kinds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up in public Hermes docs/code.",
            },
            "scope": {
                "type": "string",
                "enum": ["docs", "code", "docs_and_code"],
                "description": "docs = public docs/root docs; code = selected public upstream source; docs_and_code = both.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of cited matches to return (1-10).",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    },
}


registry.register(
    name="hermes_docs_lookup",
    toolset="docs",
    schema=HERMES_DOCS_LOOKUP_SCHEMA,
    handler=lambda args, **kw: hermes_docs_lookup_tool(
        args.get("query", ""),
        scope=args.get("scope", "docs"),
        limit=args.get("limit", DEFAULT_LIMIT),
    ),
    emoji="📚",
    max_result_size_chars=50_000,
)
