"""Run-based recording: identity, per-chunk SQLite, events, FIF export.

One SQLite file per run holds every raw chunk (data BLOB + row count + first
timestamp), task events and metadata. Chunks are committed one by one so a
killed process never loses already-written data. ``close()`` locks the run and
optionally exports the whole run as an MNE ``.fif`` next to the database.
Per-sample timestamps are NOT stored: replay rebuilds a uniform grid from each
chunk's first timestamp and row count (see ``replay_chunks``).
"""

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

# Dots and separators are stripped so a name can never escape the run tree.
_SAFE_NAME = str.maketrans("", "", "./\\:?*\"<>| \t\n")


class RunSpec:
    """Identity of one recording: subject / session / run / role.

    Args:
        subject: Subject identifier, e.g. ``"s1"``.
        session: Session identifier, e.g. ``"a"``.
        run: Run identifier within the session, e.g. ``"trial01"``.
        role: Purpose of the run, e.g. ``"trial"``, ``"baseline"``,
            ``"artifact_calibration"``.

    Notes:
        Components must be single, safe path segments; invalid characters are
        stripped rather than rejected so names survive file-system rules.
    """

    def __init__(
        self,
        subject: str,
        session: str,
        run: str,
        role: str = "run",
    ) -> None:
        """Validate and sanitize every identity component."""

        for name, value in (
            ("subject", subject),
            ("session", session),
            ("run", run),
            ("role", role),
        ):
            clean = str(value).translate(_SAFE_NAME)
            if not clean:
                raise ValueError(f"{name} must be a non-empty name.")
            setattr(self, name, clean)

    @property
    def directory(self) -> tuple[str, str, str]:
        """Directory components under the recording root."""

        return self.subject, self.session, self.run

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RunSpec(subject={self.subject!r}, session={self.session!r}, "
            f"run={self.run!r}, role={self.role!r})"
        )


class RunRecorder:
    """Record raw chunks, events and metadata into one SQLite run file.

    Args:
        root: Parent directory; the run lands in
            ``root/subject/session/run/``.
        spec: Run identity.
        channels: Channel names in column order.
        sfreq: Source sampling rate in Hz.
        ch_types: MNE channel types; defaults to all ``"eeg"``.
        unit_exponent: Power of ten of the recorded units (0 = volts).
        dtype: Storage dtype for chunk data; ``"float32"`` halves the file
            size with negligible precision loss for raw EEG.
        export_fif: Write ``<run>_raw.fif`` next to the database on ``close``.
        config: Optional serializable description of the whole run (rates,
            filters, geometry). Stored verbatim as the ``"config"`` metadata
            key so a later replay knows what produced this run.
        files: Optional ``{role: path}`` provenance inputs (e.g. the selected
            artifact operator or baseline file). Each file is copied into the
            run folder and its original path + SHA-256 are stored, so the run
            is auditable and reproducible.
        track_windows: Create the ``windows`` table and enable
            ``log_window()``; used for per-window event logging.

    Notes:
        ``write()`` commits one transaction per chunk, so a crash never loses
        already-acknowledged data. ``mark()`` is thread-safe and meant to be
        called from a task/controller thread. Calling ``close()`` more than
        once is a no-op. Use the module-level ``iter_chunks``/``iter_events``/
        ``iter_windows``/``read_metadata`` to read a run back.

    Attributes:
        samples: Rows recorded so far.
        chunks: Chunks recorded so far.
        track_windows: Whether ``log_window()`` is enabled for this run.
    """

    def __init__(
        self,
        root: str | Path,
        spec: RunSpec,
        channels: tuple[str, ...],
        sfreq: float,
        *,
        ch_types: tuple[str, ...] | None = None,
        unit_exponent: int = 0,
        dtype: type = np.float32,
        export_fif: bool = True,
        config: dict | None = None,
        files: dict | None = None,
        track_windows: bool = False,
    ) -> None:
        """Create the run directory and an empty database."""

        root = Path(root)
        if not channels or any(not str(name).strip() for name in channels):
            raise ValueError("channels must be non-empty strings.")
        if sfreq <= 0:
            raise ValueError("sfreq must be positive.")
        if ch_types is None:
            ch_types = ("eeg",) * len(channels)
        if len(ch_types) != len(channels):
            raise ValueError("ch_types must match the channel count.")
        if unit_exponent not in (0, -3, -6, -9):
            raise ValueError("Supported unit exponents are 0, -3, -6, -9.")

        # Resolve provenance inputs up front: a missing file fails the run
        # before any database or directory is created.
        provenance = []
        for role, source in (files or {}).items():
            role = str(role).translate(_SAFE_NAME)
            if not role:
                raise ValueError("Provenance roles must be non-empty names.")
            path = Path(source)
            if not path.is_file():
                raise ValueError(f"Provenance input does not exist: {path}")
            stored = self._snapshot_name(role, path)
            provenance.append((role, path, stored))

        subject, session, run = spec.directory
        self.directory = root / subject / session / run
        self.directory.mkdir(parents=True, exist_ok=True)

        # One database per run; never silently overwrite an existing one.
        self.path = self.directory / f"{run}.sqlite"
        if self.path.exists():
            raise FileExistsError(
                f"Run already exists at {self.path}; choose another run name."
            )
        self.fif_path = self.directory / f"{run}_raw.fif"

        self._spec = spec
        self._channels = tuple(str(name) for name in channels)
        self._ch_types = tuple(ch_types)
        self._sfreq = float(sfreq)
        self._unit_exponent = unit_exponent
        self._dtype = np.dtype(dtype)
        self._export_fif = export_fif
        self.samples = 0
        self.chunks = 0
        self._closed = False
        self.track_windows = bool(track_windows)
        # Serialize writes because mark() may come from another thread.
        self._lock = threading.Lock()

        # A single writer connection; check_same_thread=False is safe because
        # every access goes through self._lock.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE chunks("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            "n_samples INTEGER NOT NULL,"
            "first_timestamp REAL,"
            "data BLOB NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE events("
            "rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
            "ts REAL NOT NULL,"
            "label TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.execute("INSERT INTO meta VALUES ('status', 'recording')")
        self._set_meta("channels", json.dumps(list(self._channels)))
        self._set_meta("ch_types", json.dumps(list(self._ch_types)))
        self._set_meta("sfreq", str(self._sfreq))
        self._set_meta("unit_exponent", str(self._unit_exponent))
        self._set_meta("dtype", np.dtype(dtype).str)
        self._set_meta("subject", spec.subject)
        self._set_meta("session", spec.session)
        self._set_meta("run", spec.run)
        self._set_meta("role", spec.role)
        if config is not None:
            self._set_meta("config", json.dumps(config))
        if track_windows:
            self._connection.execute(
                "CREATE TABLE windows("
                "rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
                "ts REAL NOT NULL,"
                "valid INTEGER NOT NULL,"
                "reasons TEXT,"
                "segment INTEGER NOT NULL,"
                "artifact_id TEXT)"
            )
        self._connection.commit()

        # Copy each provenance input into the run folder and record its hash.
        for role, source, stored in provenance:
            destination = self.directory / stored
            digest = self._copy_and_hash(source, destination)
            self._set_meta(
                f"input:{role}",
                json.dumps(
                    {
                        "original": str(source),
                        "sha256": digest,
                        "stored": stored,
                    }
                ),
            )
        self._connection.commit()

    @staticmethod
    def _snapshot_name(role: str, source: Path) -> str:
        """Build a safe stored filename for a provenance input."""

        suffix = source.suffix if source.suffix else ".bin"
        return f"{role}{suffix}"

    @staticmethod
    def _copy_and_hash(source: Path, destination: Path) -> str:
        """Copy a file while hashing it, refusing to overwrite anything."""

        import hashlib

        if destination.exists():
            raise FileExistsError(f"Snapshot already exists: {destination}")
        hasher = hashlib.sha256()
        with source.open("rb") as src, destination.open("xb") as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                hasher.update(block)
                dst.write(block)
        return hasher.hexdigest()

    def _set_meta(self, key: str, value: str) -> None:
        """Upsert one metadata key without touching the commit boundary."""

        self._connection.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def write(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Commit one raw chunk (samples by channels) immediately."""

        if self._closed:
            raise RuntimeError("This RunRecorder is closed.")
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != len(self._channels):
            raise ValueError(
                f"Expected samples by {len(self._channels)} channels, "
                f"received shape {data.shape}."
            )
        if timestamps.ndim != 1 or len(timestamps) != len(data):
            raise ValueError("Each sample must have one timestamp.")
        if len(data) == 0:
            return

        # Keep the first real time of the chunk; per-sample times are rebuilt
        # later from sfreq.
        finite = timestamps[np.isfinite(timestamps)]
        first = float(finite[0]) if finite.size else None

        blob = np.ascontiguousarray(data, dtype=self._dtype).tobytes()
        with self._lock:
            self._connection.execute(
                "INSERT INTO chunks(n_samples, first_timestamp, data) "
                "VALUES (?, ?, ?)",
                (len(data), first, blob),
            )
            self._connection.commit()
        self.samples += len(data)
        self.chunks += 1

    def mark(self, label: str, timestamp: float | None = None) -> None:
        """Record one task event from any thread.

        Args:
            label: Event name, e.g. ``"blink"`` or ``"stimulus"``.
            timestamp: Original event time; defaults to the local LSL clock if
                ``None``. Use the same clock the source timestamps use.
        """

        if self._closed:
            raise RuntimeError("This RunRecorder is closed.")
        if not str(label).strip():
            raise ValueError("label must be a non-empty string.")

        if timestamp is None:
            from mne_lsl.lsl import local_clock

            timestamp = local_clock()

        with self._lock:
            self._connection.execute(
                "INSERT INTO events(ts, label) VALUES (?, ?)",
                (float(timestamp), str(label)),
            )
            self._connection.commit()

    def mark_not_processed(
        self, rows: int, timestamp: float | None = None
    ) -> None:
        """Record rows that were still buffered when the run stopped.

        The real-time loop can stop with a tail of samples still in the
        acquire handle (or the ring). Those rows were never processed and
        never delivered; marking them keeps the run's bookkeeping honest
        instead of silently dropping them.

        Args:
            rows: Number of buffered, unprocessed rows.
            timestamp: Stop time; defaults to the local LSL clock.
        """

        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError("rows must be a positive integer.")
        if timestamp is None:
            from mne_lsl.lsl import local_clock

            timestamp = local_clock()

        with self._lock:
            self._connection.execute(
                "INSERT INTO events(ts, label) VALUES (?, ?)",
                (float(timestamp), f"not_processed:{int(rows)}"),
            )
            self._connection.commit()

    def log_window(
        self,
        timestamp: float,
        valid: bool,
        reasons: tuple[str, ...] = (),
        segment: int = 0,
        artifact_id: str | None = None,
    ) -> None:
        """Record one delivered window when window tracking is enabled.

        Args:
            timestamp: Delivery time of the window (source clock).
            valid: Whether the window passed the gate.
            reasons: Rejection reasons of the window.
            segment: Processing segment the window came from.
            artifact_id: Applied spatial operator, when one was used.
        """

        if not self.track_windows:
            raise RuntimeError("This run does not track windows.")
        if self._closed:
            raise RuntimeError("This RunRecorder is closed.")

        with self._lock:
            self._connection.execute(
                "INSERT INTO windows(ts, valid, reasons, segment, artifact_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    float(timestamp),
                    int(bool(valid)),
                    json.dumps(list(reasons)),
                    int(segment),
                    artifact_id,
                ),
            )
            self._connection.commit()

    def close(
        self,
        status: str = "completed",
        error: str | None = None,
        stats: dict | None = None,
    ) -> Path:
        """Lock the run, export the FIF if asked, and return the database path.

        Args:
            status: ``"completed"`` or ``"failed"``.
            error: Optional human-readable error for failed runs.
            stats: Optional serializable counters to store in the metadata.
        """

        if self._closed:
            return self.path
        self._closed = True

        with self._lock:
            self._set_meta("status", status)
            self._set_meta("samples", str(self.samples))
            self._set_meta("chunks", str(self.chunks))
            if error is not None:
                self._set_meta("error", error)
            if stats is not None:
                self._set_meta("stats", json.dumps(stats))
            self._connection.commit()

        # The FIF export reads the database back once at the end. Close the
        # connection even if the export fails, so a failed write cannot leak
        # the handle (or mask the original error behind a locked file).
        try:
            if self._export_fif and self.chunks:
                self._export_fif_file()
        finally:
            self._connection.close()

        # Human-readable sidecar next to the database.
        metadata = read_metadata(self.path)
        self.path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return self.path

    def _export_fif_file(self) -> None:
        """Write the whole run as an MNE Raw FIF in source units."""

        import mne

        blocks = [data for data, _ in iter_chunks(self.path)]
        signal = np.concatenate([block.T for block in blocks], axis=1)
        info = mne.create_info(
            list(self._channels), self._sfreq, self._ch_types, verbose=False
        )
        raw = mne.io.RawArray(np.asarray(signal, dtype=np.float64), info, verbose=False)
        raw.save(self.fif_path, overwrite=True, verbose=False)

    @classmethod
    def read_metadata(cls, path: str | Path) -> dict:
        """Read the metadata table of a recorded run."""

        return read_metadata(path)


def _read_connection(path: str | Path) -> sqlite3.Connection:
    """Open a read-only connection, POSIX paths keep the URI well formed."""

    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def read_metadata(path: str | Path) -> dict:
    """Return every metadata key of a run, JSON-decoding structured values."""

    connection = _read_connection(path)
    try:
        rows = connection.execute("SELECT key, value FROM meta").fetchall()
    finally:
        connection.close()

    metadata = {}
    for key, value in rows:
        try:
            metadata[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            metadata[key] = value
    return metadata


def iter_chunks(path: str | Path) -> "Iterator[tuple[np.ndarray, float | None]]":
    """Yield every recorded chunk as ``(data, first_timestamp)`` in order.

    ``data`` is samples by channels in the stored dtype. Rebuild a timestamp
    grid with ``chunk_timestamps`` if needed.
    """

    connection = _read_connection(path)
    try:
        metadata = read_metadata(path)
        dtype = np.dtype(metadata.get("dtype", "<f4"))
        rows = connection.execute(
            "SELECT n_samples, first_timestamp, data FROM chunks ORDER BY seq"
        )
        for n_samples, first_timestamp, blob in rows:
            data = np.frombuffer(blob, dtype=dtype).reshape(int(n_samples), -1)
            yield data, first_timestamp
    finally:
        connection.close()


def iter_events(path: str | Path) -> list[tuple[float, str]]:
    """Return recorded events as ``(timestamp, label)`` in arrival order."""

    connection = _read_connection(path)
    try:
        rows = connection.execute("SELECT ts, label FROM events ORDER BY rowid")
        return [(float(ts), str(label)) for ts, label in rows]
    finally:
        connection.close()


def iter_windows(
    path: str | Path,
) -> list[tuple[float, bool, tuple[str, ...], int, str | None]]:
    """Return logged windows as ``(ts, valid, reasons, segment, artifact_id)``.

    Runs recorded without window tracking have no ``windows`` table; reading
    them yields an empty list.
    """

    connection = _read_connection(path)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='windows'"
        ).fetchone()
        if exists is None:
            return []
        rows = connection.execute(
            "SELECT ts, valid, reasons, segment, artifact_id "
            "FROM windows ORDER BY rowid"
        )
        return [
            (
                float(ts),
                bool(valid),
                tuple(json.loads(reasons)) if reasons else (),
                int(segment),
                artifact_id,
            )
            for ts, valid, reasons, segment, artifact_id in rows
        ]
    finally:
        connection.close()


def chunk_timestamps(
    first_timestamp: float | None,
    n_samples: int,
    sfreq: float,
) -> np.ndarray:
    """Rebuild a chunk's uniform timestamp grid from its anchor."""

    if first_timestamp is None:
        return np.full(n_samples, np.nan)
    return first_timestamp + np.arange(n_samples) / sfreq


def replay_chunks(
    path: str | Path,
) -> "Iterator[tuple[np.ndarray, np.ndarray]]":
    """Yield every chunk with rebuilt timestamps for offline re-processing.

    This is the entry point for feeding a recorded run back through the same
    preprocessing chain later (offline evaluation before any model exists).
    """

    metadata = read_metadata(path)
    sfreq = float(metadata.get("sfreq", 0.0))
    for data, first_timestamp in iter_chunks(path):
        timestamps = chunk_timestamps(first_timestamp, len(data), sfreq)
        yield data, timestamps
