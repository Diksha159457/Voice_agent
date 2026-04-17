# utils/memory.py — Session Memory module
# Lightweight in-memory history store. No database, no files, no dependencies.
# History is lost on server restart — acceptable for a demo/single-user app.
# To persist across restarts, swap _memory for a SQLite or JSON-file backend.

_memory: list = []   # module-level list; shared across all requests in this process

MAX_ENTRIES = 100    # hard cap — prevents unbounded RAM growth on long sessions


def add_to_memory(entry: dict) -> None:
    """
    Append one interaction to history and trim if over the cap.

    Args:
        entry: dict with keys 'text', 'intent', 'result'
    """
    _memory.append(entry)           # add newest at the end (chronological order)
    if len(_memory) > MAX_ENTRIES:
        _memory.pop(0)              # drop oldest entry — keeps list at MAX_ENTRIES


def get_memory() -> list:
    """Return a shallow copy of the history list (oldest first)."""
    return list(_memory)            # copy prevents callers mutating _memory directly


def clear_memory() -> None:
    """Wipe all history. Called by the Clear History button."""
    _memory.clear()                 # empties in-place; any other reference also sees []