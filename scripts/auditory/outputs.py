"""Refuse to silently replace a previous run's outputs.

Every auditory CLI writes a small, fixed set of files into ``--out``. Re-running
with the same path is usually a mistake, and without this guard it destroys the
previous session with no warning. The write is therefore refused unless the
operator asks for it explicitly with ``--force``.
"""

from pathlib import Path


def guard_outputs(paths, *, force=False):
    """Raise if any of ``paths`` already exists and ``force`` was not given.

    Args:
        paths: The files the caller is about to write.
        force: Overwrite the existing files instead of refusing.

    Raises:
        FileExistsError: If a target exists and ``force`` is false.
    """

    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to replace existing output ("
            + ", ".join(existing)
            + "); pass --force to overwrite."
        )
