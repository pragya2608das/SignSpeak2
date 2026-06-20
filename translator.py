WINDOW_SIZE = 32
STRIDE = 16


def create_windows(sequence):

    windows = []

    if len(sequence) < WINDOW_SIZE:
        return []

    for start in range(
        0,
        len(sequence) - WINDOW_SIZE + 1,
        STRIDE
    ):

        end = start + WINDOW_SIZE

        windows.append(
            sequence[start:end]
        )

    return windows


def remove_duplicates(words):

    output = []

    for word in words:

        if len(output) == 0:

            output.append(word)

        elif output[-1] != word:

            output.append(word)

    return output