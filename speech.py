import pyttsx3

engine = pyttsx3.init()

engine.setProperty(
    "rate",
    160
)

engine.setProperty(
    "volume",
    1.0
)

voices = engine.getProperty(
    "voices"
)

if len(voices) > 0:

    engine.setProperty(
        "voice",
        voices[0].id
    )

def speak(text):

    if not text:
        return

    engine.say(text)

    engine.runAndWait()