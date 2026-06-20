from translator import (
    create_windows,
    remove_duplicates
)


def translate_video(
    sequence,
    predictor
):

    windows = create_windows(
        sequence
    )

    print(
        f"Windows Created: {len(windows)}"
    )

    words = []

    for window in windows:

        word, conf = (
            predictor
            .predict_word_with_confidence(
                window
            )
        )

        print(
            word,
            f"{conf:.2f}%"
        )

        if conf > 25:

            words.append(
                word
            )

    words = remove_duplicates(
        words
    )

    return words