import pyttsx3
import threading

is_speaking = False


def _speak(text):
    global is_speaking

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    is_speaking = True

    engine.say(text)
    engine.runAndWait()
    engine.stop()

    is_speaking = False


def speak(text):
    global is_speaking

    if text.strip() == "":
        return

    # Don't start another speech if one is already playing
    if is_speaking:
        return

    threading.Thread(
        target=_speak,
        args=(text,),
        daemon=True
    ).start()