"""Small helpers for matching OmniNav's bounded visual-history input."""


def policy_history(images, window):
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("history window must be a nonnegative integer")
    return images[:-1][-window:] if window else []


def append_policy_frame(images, image, window):
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("history window must be a nonnegative integer")
    images.append(image)
    if window == 0:
        images[:] = images[-1:]
    elif len(images) > window + 1:
        images[:] = [images[0], *images[-window:]]
    return images
