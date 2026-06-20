from webcam_processor import (
    get_sequence,
    add_word,
    get_words
)

def process_sign(
    predictor
):

    sequence = get_sequence()

    if sequence is None:

        return None

    word = predictor.predict_word(
        sequence
    )

    add_word(word)

    return word


def get_sentence_words():

    return get_words()