"""Select prior observations in the order expected by the frozen checkpoint."""


def native_history_from_full(frames, window):
    if type(window) is not int or window < 0:
        raise ValueError("history window must be a nonnegative integer")
    if window == 0:
        return []
    if len(frames) <= window + 1:
        return frames[:-1]
    # Keep the initial observation, then recent history; exclude the current frame.
    return [frames[0], *frames[-window:-1]]
