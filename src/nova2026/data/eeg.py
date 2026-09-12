from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np
import torch

from nova2026.config import DATA_DIR, SAMPLE_SIZE
from nova2026.data.pipeline import Pipeline, PipelineError


def _gen_output_path(
    data_type: str, pipeline_name: str, sample_rate: float, output_dir: Path
) -> Path:
    return output_dir / f"{data_type}_{sample_rate}Hz_{pipeline_name}.pt"


def save_chkpt(chkpt, output_dir) -> None:
    output_path: Path = _gen_output_path(
        chkpt["data_type"], chkpt["pipeline"], chkpt["sample_rate_hz"], output_dir
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(chkpt, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_eeg(set_file: str | Path) -> mne.io.BaseRaw:
    """Load an EEGLAB ``.set`` file into a raw MNE object.

    Args:
        set_file (str | Path): Path to the ``.set`` file to load.

    Returns:
        mne.io.BaseRaw: The loaded raw data.
    """
    return mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)


"""Supported formats of EEG recording files.

The list holds the file suffixes (e.g. ``".set"``) that are recognized
when searching with ``query_type="suffix"``.
"""
EEG_DATA_FORMAT = [".set", ".vhdr", ".dat", ".edf", ".bdf", ".cnt", ".txt", ".raw"]

LOAD_OPTS = {
    "eeglab": mne.io.read_raw_eeglab,
    "brainvision": mne.io.read_raw_brainvision,
    "edf": mne.io.read_raw_edf,
    "bdf": mne.io.read_raw_bdf,
    "cnt": mne.io.read_raw_cnt,
}


class Loader:
    """Recursively searches a dataset directory for EEG recording files.

    Args:
        dataset (str): Name of the dataset subfolder under the root directory.
        root (Path | None, optional): Base directory containing the dataset.
            Defaults to ``DATA_DIR``.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """

    class DataFileRef:
        """Represents a reference to a data file.

        Attributes:
            path (Path): The path to the data file.
            tags (list[str] | None): Optional tags associated with the data file.
        """

        def __init__(self, path: Path, tags: list[str] | None) -> None:
            self.path: Path = path
            self.tags: list[str] | None = tags

        def __str__(self) -> str:
            return f"[{self.tags}: {self.path}]"

        def __repr__(self) -> str:
            return self.__str__()

    class TaggedData:
        """Represents a tagged raw data object.
        Defaults loads the data from the file during initializing.

        Attributes:
            raw (mne.io.BaseRaw | None): The raw data object.
            tags (list[str] | None): Optional tags associated with the data.
        """

        def __init__(self, raw: mne.io.BaseRaw, tags: list[str]) -> None:
            self.raw: mne.io.BaseRaw = raw
            self.raw.load_data()
            self.tags: list[str] = tags

        def is_tagged_with(self, tag: str) -> bool:
            return tag in self.tags

    def __init__(self, dataset: str, root: Path | None = None):
        """Initialize the Loader.

        Args:
            dataset (str): Name of the dataset subfolder under the root directory.
            root (Path | None, optional): Base directory containing the dataset.
                Defaults to ``DATA_DIR``.

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
        """
        self.root: Path = DATA_DIR if root is None else root
        self.dataset: Path = self.root / dataset
        self.query_cache: list[Loader.DataFileRef] = []
        self.tagged_data: list[Loader.TaggedData] = []
        if not (self.dataset).exists():
            raise FileNotFoundError(f"Dataset {self.dataset} not found in {self.root}")

    def clear_query(self):
        """Clear the cached search results."""
        self.query_cache = []

    def clear_data(self):
        self.tagged_data = []

    def clear(self):
        self.clear_query()
        self.clear_data()

    def search(
        self, query: str, query_type: str, query_path: Path | None = None
    ) -> list[DataFileRef]:
        """Recursively search for files matching a name or suffix query.

        Args:
            query (str): The filename stem (for ``query_type="name"``) or a
                file format like ``".set"`` (for ``query_type="suffix"``).
            query_type (str): Either ``"name"`` or ``"suffix"``.
            query_path (Path | None, optional): Directory to start the search
                from. Defaults to the dataset directory.

        Returns:
            list[DataFileRef]: References to all matching files.

        Raises:
            ValueError: If ``query_type`` is ``"suffix"`` and the query is not
                in :data:`EEG_DATA_FORMAT`.
        """
        query_path = query_path if query_path is not None else self.dataset

        # suffix check
        if query_type == "suffix" and not (query in EEG_DATA_FORMAT):
            raise ValueError(f"Unsupported file format: {query}")

        for item_path in query_path.iterdir():
            # Recursive search
            if item_path.is_dir():
                self.search(query, query_type, item_path)
                continue

            # check if the file matches the query
            hit_by_name = (query_type == "name") and (query in item_path.stem)
            hit_by_suffix = (query_type == "suffix") and (query in item_path.suffix)
            if hit_by_name or hit_by_suffix:
                self.query_cache.append(Loader.DataFileRef(item_path, None))

        return self.query_cache

    def refine(self, validator: Callable[[DataFileRef], bool]):
        """Filter the cached search results by a predicate.

        Args:
            validator (Callable[[DataFileRef], bool]): Predicate deciding
                whether a cached reference is kept.

        Returns:
            list[DataFileRef]: The filtered subset of the cached results.
        """
        self.query_cache = [p for p in self.query_cache if validator(p)]
        return self.query_cache

    def tag(self, tagger: Callable[[DataFileRef], list[str]]):
        """Tag the cached search results.

        Args:
            tagger (Callable[[DataFileRef], list[str]]): Function that takes
                a data file reference and returns a list of tags.

        Returns:
            list[DataFileRef]: The tagged subset of the cached results.
        """
        for q in self.query_cache:
            q.tags = tagger(q)
        return self.query_cache

    def load(
        self,
        mode: str | None = None,
        loader: Callable[[DataFileRef], TaggedData] | None = None,
    ) -> list[TaggedData]:
        """Load the cached search results.

        Args:
            mode (str | None, optional): Name of a predefined loader in
                :data:`LOAD_OPTS` (e.g. ``"eeglab"``). Mutually exclusive
                with ``loader``.
            loader (Callable[[DataFileRef], TaggedData] | None, optional):
                Custom function that takes a DataFileRef object and returns
                a TaggedData object. Defaults to ``None``.

        Returns:
            list[TaggedData]: The loaded subset of the cached results.

        Raises:
            ValueError: If neither ``mode`` nor ``loader`` is given, or if
                ``mode`` is not a key of :data:`LOAD_OPTS`.
        """
        loader_func = None
        if loader is None:
            if mode is None:
                raise ValueError("Either mode or loader must be specified")
            else:
                mne_func = LOAD_OPTS.get(mode)
                if mne_func is None:
                    raise ValueError(f"Unknown mode: {mode}")
                loader_func = lambda drf: Loader.TaggedData(
                    raw=mne_func(drf.path, verbose=False), tags=drf.tags
                )
        else:
            loader_func = loader

        if loader_func is None:
            raise ValueError("Unable to obtain loader.")

        loaded_list: list[Loader.TaggedData] = []
        for q in self.query_cache:
            loaded_list.append(loader_func(q))
        self.tagged_data = loaded_list
        return loaded_list

    def run_pipe(self, pipeline: Pipeline):
        if len(self.tagged_data) == 0:
            raise ValueError("No data to run pipeline on.")
        for tagged_data in self.tagged_data:
            _, result_raw = pipeline.rundown(tagged_data.raw)
            if not isinstance(result_raw, mne.io.BaseRaw):
                raise PipelineError("Pipeline must return a mne.io.BaseRaw object.")
            tagged_data.raw = result_raw

    def run(self, func: Callable[[mne.io.BaseRaw], mne.io.BaseRaw]):
        """Run a function on all tagged data.

        Args:
            func (Callable[[mne.io.BaseRaw], mne.io.BaseRaw]): The function to
                run on each raw object.
        """
        for tagged_data in self.tagged_data:
            result_raw = func(tagged_data.raw)
            if not isinstance(result_raw, mne.io.BaseRaw):
                raise PipelineError("Pipeline must return a mne.io.BaseRaw object.")
            tagged_data.raw = result_raw

    def slice(
        self,
        cuts_list: list[list[int]],
        eeg_channels: list[str],
        sample_size: int = SAMPLE_SIZE,
        cut_at_start: bool = True,
        permutate: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[list[list[str]], list[np.ndarray]]:
        """Slicing the tagged data with given cuts.

        Args:
            cuts_list (list[list[int]]): List of cut-index lists. ``cuts_list[i]``
                gives the cut indices for ``self.tagged_data[i]``, so its length
                must equal the number of tagged data.
            eeg_channels (list[str]): List of EEG channel names.
            sample_size (int, optional): Size of each window. Defaults to
                :data:`SAMPLE_SIZE`.
            permutate (Callable[[np.ndarray], np.ndarray] | None, optional):
                Function applied to each window before it is stored. If given,
                ``data_list[i]`` is ``permutate(window)``. The function must
                return a new array: the window passed to it is a view of the
                recording, so it must not be modified in place.
            cut_at_start (bool, optional): If ``True``, each window starts at
                ``cut``; otherwise it ends at ``cut``.

        Returns:
            tuple[list[list[str]], list[np.ndarray]]: A pair ``(meta_list,
            data_list)``. Each list has one entry per window: ``meta_list[i]``
            holds the tags of the recording that window came from, and
            ``data_list[i]`` is the ``(n_channels, sample_size)`` window (after
            ``permutate``, if provided).

        Raises:
            TypeError: If the EEG recording is not a numpy array.
            ValueError: If there is no tagged data, ``cuts_list`` length does
                not match the tagged data, a cuts list is empty, or a cut
                index is out of bounds.
        """

        if len(self.tagged_data) == 0:
            raise ValueError("No data to slice.")

        meta_list: list[list[str]] = []
        data_list: list[np.ndarray] = []

        if len(self.tagged_data) != len(cuts_list):
            raise ValueError("Number of cuts lists must match number of tagged data.")

        for tagged_data, cuts in zip(self.tagged_data, cuts_list):
            recording = tagged_data.raw.get_data(picks=eeg_channels)
            if not isinstance(recording, np.ndarray):
                raise TypeError("EEG recording must be a numpy array.")

            cuts = sorted(cuts)
            n_times = recording.shape[1]
            if len(cuts) == 0:
                raise ValueError("No cuts provided.")
            if cut_at_start:
                if cuts[0] < 0 or cuts[-1] + sample_size > n_times:
                    raise ValueError("Cut indices out of bounds.")
            else:
                if cuts[0] - sample_size < 0 or cuts[-1] > n_times:
                    raise ValueError("Cut indices out of bounds.")

            for cut in cuts:
                window: np.ndarray
                if cut_at_start:
                    window = recording[:, cut : cut + sample_size]
                else:
                    window = recording[:, cut - sample_size : cut]
                if permutate is None:
                    data_list.append(window)
                else:
                    data_list.append(permutate(window))
                meta_list.append(list(tagged_data.tags))

        return meta_list, data_list

    def slice_const_interval(
        self,
        eeg_channels: list[str],
        sample_size: int = SAMPLE_SIZE,
        trim: tuple[int, int] | int = 0,
        hop: int = -1,
        permutate: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[list[list[str]], list[np.ndarray]]:
        """Slicing the tagged data with constant interval.

        Args:
            eeg_channels (list[str]): List of EEG channel names.
            sample_size (int, optional): Size of each window. Defaults to
                :data:`SAMPLE_SIZE`.
            trim (tuple[int, int] | int, optional): Number of samples to trim
                from the beginning and end of the recording. An ``int`` value
                is applied to both ends. Defaults to ``0`` (no trimming).
            hop (int, optional): Hop size between windows. Defaults to -1,
                which means hop = sample_size.
            permutate (Callable[[np.ndarray], np.ndarray] | None, optional):
                Function applied to each window before it is stored. If given,
                ``data_list[i]`` is ``permutate(window)``. The function must
                return a new array: the window passed to it is a view of the
                recording, so it must not be modified in place.

        Returns:
            tuple[list[list[str]], list[np.ndarray]]: A pair ``(meta_list,
            data_list)``. Each list has one entry per window: ``meta_list[i]``
            holds the tags of the recording that window came from, and
            ``data_list[i]`` is the ``(n_channels, sample_size)`` window (after
            ``permutate``, if provided).

        Raises:
            TypeError: If the EEG recording is not a numpy array.
            ValueError: If there is no tagged data, the trim leaves no data,
                or the trimmed recording is shorter than ``sample_size``.
        """

        if hop <= 0:
            hop = sample_size

        if len(self.tagged_data) == 0:
            raise ValueError("No data to slice.")

        if isinstance(trim, int):
            trim = (trim, trim)

        meta_list: list[list[str]] = []
        data_list: list[np.ndarray] = []
        for tagged_data in self.tagged_data:
            recording = tagged_data.raw.get_data(picks=eeg_channels)
            if not isinstance(recording, np.ndarray):
                raise TypeError("EEG recording must be a numpy array")

            start, end = trim[0], recording.shape[1] - trim[1]
            if start > end:
                raise ValueError(f"Trim start {start} is greater than trim end {end}.")

            recording = recording[:, start:end]
            if recording.shape[1] < sample_size:
                raise ValueError(
                    f"Recording length {recording.shape[1]} after trim is "
                    f"shorter than sample_size {sample_size}."
                )
            for idx in range(0, recording.shape[1] - sample_size + 1, hop):
                if permutate is None:
                    data_list.append(recording[:, idx : idx + sample_size])
                else:
                    data_list.append(permutate(recording[:, idx : idx + sample_size]))
                meta_list.append(list(tagged_data.tags))

        return meta_list, data_list
