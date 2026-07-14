class WordBuilder:

    def __init__(self):

        self.current_word = ""
        self.current_sentence = ""

        self.last_prediction = ""
        self.hold_frames = 0

        self.letter_added = False

        self.HOLD_THRESHOLD = 20

    def update(self, prediction):

        # Same prediction continues
        if prediction == self.last_prediction:
            self.hold_frames += 1

        else:
            self.last_prediction = prediction
            self.hold_frames = 0
            self.letter_added = False

        # Accept only once
        if (
            self.hold_frames >= self.HOLD_THRESHOLD
            and not self.letter_added
        ):

            if prediction == "space":

                if self.current_word != "":
                    self.current_sentence += self.current_word + " "
                    self.current_word = ""

            elif prediction == "del":

                self.current_word = self.current_word[:-1]

            else:

                self.current_word += prediction

            self.letter_added = True

    def get_progress(self):

        return int(
            min(self.hold_frames, self.HOLD_THRESHOLD)
            / self.HOLD_THRESHOLD
            * 250
        )

    def clear(self):

        self.current_word = ""
        self.current_sentence = ""

    def reset_prediction(self):

        self.last_prediction = ""
        self.hold_frames = 0
        self.letter_added = False

    def get_word(self):

        return self.current_word

    def get_sentence(self):

        return self.current_sentence