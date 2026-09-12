"""The window-judge protocol: a verdict plus the evidence behind it.

A judge answers two questions about a finished window's time span:

* ``reasons(start, end) -> tuple[str, ...]`` — required. A non-empty answer
  rejects the window.
* ``bad_channels(start, end) -> tuple[str, ...]`` — optional. Which EEG
  channels the judge found faulty. This is evidence, never a verdict: a run
  configured to tolerate a dead electrode still records that it was dead.

Both the session's window gate and the auditory chain's window loop ask judges
through :func:`collect_verdict`, so a judge is one small object either way and
adding a new one never means editing a core loop.
"""


def judge_reasons(judge, start: float, end: float) -> tuple[str, ...]:
    """Ask one judge for its rejection reasons."""

    method = getattr(judge, "reasons", None)
    if not callable(method):
        raise TypeError(
            "Every window judge must implement reasons(start, end); "
            f"{judge!r} does not."
        )
    return tuple(str(reason) for reason in method(start, end))


def judge_bad_channels(judge, start: float, end: float) -> tuple[str, ...]:
    """Ask one judge for its bad-channel census, when it reports one.

    The method is optional on purpose: a judge written before bad-channel
    reporting existed keeps working unchanged, and a judge with no channel
    concept simply contributes nothing.
    """

    method = getattr(judge, "bad_channels", None)
    if not callable(method):
        return ()
    return tuple(str(name) for name in method(start, end))


def collect_verdict(
    judges, start: float, end: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Union every judge over one time span.

    Returns:
        ``(reasons, bad_channels)``, both sorted and de-duplicated. Reasons
        reject the window; bad channels only describe it.

    Raises:
        TypeError: When a judge does not implement ``reasons(start, end)``.
    """

    reasons: set[str] = set()
    bad_channels: set[str] = set()
    for judge in judges:
        reasons.update(judge_reasons(judge, start, end))
        bad_channels.update(judge_bad_channels(judge, start, end))

    return tuple(sorted(reasons)), tuple(sorted(bad_channels))
