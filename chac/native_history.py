from .history import policy_history as RAW_HISTORY


def native_history_from_full(images, window):
    prior = RAW_HISTORY(
        images, window
    )  # Also validates window, including bool and zero.
    if window and len(images) > window + 1:
        return RAW_HISTORY([images[0], *images[-window:]], window)
    return prior
