"""Filesystem sandbox for agent-created files.

Every path the agent writes to comes (indirectly) from an LLM or a user's
speech, so it is treated as hostile. A name is accepted only if:

  * it is a single path component made of safe characters,
  * it isn't ``.``/``..``/hidden, and isn't absurdly long,
  * the fully-resolved path (after following symlinks) is still inside the
    sandbox root.

Anything else raises ``SandboxError`` instead of being silently "fixed".
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,99}$")


class SandboxError(ValueError):
    """Raised when a requested name would escape or abuse the sandbox."""


def safe_path(root: str | os.PathLike[str], name: object) -> Path:
    if not isinstance(name, str):
        raise SandboxError("File name must be text.")
    name = name.strip().strip("'\"")
    if not name:
        raise SandboxError("Please specify a file or folder name.")
    if "/" in name or "\\" in name or "\x00" in name:
        raise SandboxError(f"'{name}' must be a plain name, not a path.")
    if name in {".", ".."} or not _SAFE_NAME.match(name):
        raise SandboxError(
            f"'{name}' is not allowed. Use letters, numbers, spaces, '.', '-' or '_' (max 100 chars)."
        )

    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    candidate = (root_path / name).resolve()
    if candidate.parent != root_path:
        raise SandboxError(f"'{name}' resolves outside the output folder.")
    return candidate
