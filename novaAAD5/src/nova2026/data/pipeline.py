from collections.abc import Callable
from typing import Any


class PipelineError(Exception):
    pass


class Pipeline:
    """A series of transformation steps applied to data of a single type.

    Tubes are registered with :meth:`add_tube` and form an ordered chain of
    transformations. A Pipeline instance holds at most one piece of data
    (its "load") at a time: :meth:`feed` loads it for stepwise processing via
    :meth:`step`, while :meth:`rundown` runs the whole chain on arbitrary
    data without interfering with the load already in the pipeline.
    """

    def __init__(self):
        self.__tubes__: list[Callable[[Any], tuple[Any, Any]]] = []
        self.__load__: Any | None = None
        self.__step_loc__: int = 0

    def add_tube(self, tube: Callable[[Any], tuple[Any, Any]]):
        """Append a transformation step to the end of the chain.

        Parameters
        ----------
        tube
            Callable taking the current data and returning
            ``(result, new_data)``.
        """
        self.__tubes__.append(tube)

    def rundown(self, data: Any):
        """Run the full chain on ``data`` from start to finish.

        The given data passes through every registered tube in order. The
        pipeline's current load, if any, is left untouched.

        Parameters
        ----------
        data
            Data to process.

        Returns
        -------
        tuple
            ``(result, final_data)`` from the last tube.
        """

        if data is None:
            raise PipelineError("data is type None. Please provide actual data.")

        is_self_load = data is self.__load__

        if is_self_load:
            print(
                f"[data: {data}] is the same as self.__load__: {self.__load__}. Running the pipeline on self.__load__"
            )

        temp_load = data
        result = None
        for tube in self.__tubes__:
            result, temp_load = tube(temp_load)

        if is_self_load:
            self.__load__ = (
                temp_load  # in case pipeline doesn't modify everything in-place
            )
            return result, self.spit()  # spit resets and returns modified self.__load__

        return result, temp_load

    def feed(self, data: Any):
        """Load ``data`` into the pipeline and reset the step position.

        Parameters
        ----------
        data
            Data to hold for stepwise processing.

        Raises
        ------
        PipelineError
            If the pipeline is already holding data.
        """
        if self.__load__ is None:
            self.__load__ = data
            self.__step_loc__ = 0
        else:
            raise PipelineError(
                "Pipeline is already processing data. Call step() first."
            )

    def step(self, steps: int = 1):
        """Execute one or more tubes on the held load.

        Parameters
        ----------
        steps
            Number of tubes to run (default 1).

        Returns
        -------
        tuple
            ``(result, current_data)`` after the executed tubes.

        Raises
        ------
        PipelineError
            If there are no tubes or no data to process.
        ValueError
            If ``steps`` is not positive, or if the pipeline has no data or
            has already completed processing.
        """
        if len(self.__tubes__) == 0:
            raise PipelineError("Pipeline has no tubes to process data.")
        if steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if self.__load__ is None:
            raise ValueError("Pipeline has no data to process. Call feed() first.")
        if self.__step_loc__ >= len(self.__tubes__):
            self.__load__ = None
            self.__step_loc__ = 0
            raise ValueError(
                "Pipeline has already completed processing. Call feed() first."
            )

        result = None
        steps_forward = min(steps, len(self.__tubes__) - self.__step_loc__)

        for _ in range(steps_forward):
            result, self.__load__ = self.__tubes__[self.__step_loc__](self.__load__)
            self.__step_loc__ += 1

        load = self.__load__
        if self.__step_loc__ >= len(self.__tubes__):
            self.__load__ = None
            self.__step_loc__ = 0
        return result, load

    def spit(self):
        """Return and clear the held load.

        Returns
        -------
        D | None
            The remaining data, or ``None`` if the pipeline is empty.
        """
        load = self.__load__
        self.__step_loc__ = 0
        self.__load__ = None
        return load


from mne.io import BaseRaw

from nova2026.config import SAMPLE_RATE


class DefaultPipe(Pipeline):
    def __init__(
        self,
        l_freq=0.5,
        h_freq=45.0,
        picks="eeg",
        sample_rate=SAMPLE_RATE,
        verbose=False,
    ):
        super().__init__()

        def filter_raw_eeg(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks=picks,
                fir_design="firwin",
                verbose=verbose,
            )
            return None, raw

        def resample_raw_eeg(raw: BaseRaw) -> tuple[None, BaseRaw]:
            raw.resample(sample_rate, verbose=verbose)
            return None, raw

        self.add_tube(filter_raw_eeg)
        self.add_tube(resample_raw_eeg)
