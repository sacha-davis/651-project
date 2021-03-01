import numpy as np
import tensorflow as tf
from tensorflow import keras


# This code is largely taken from the keras github: https://github.com/keras-team/keras-io/blob/master/examples/nlp/lstm_seq2seq.py

class Seq2Seq:
    def __init__(self, batch_size, epochs, latent_dim, num_samples, data_path,
                 model_file=None):
        self.batch_size = batch_size  # Batch size for training.
        self.epochs = epochs  # Number of epochs to train for.
        self.latent_dim = latent_dim  # Latent dimensionality of the encoding space.
        self.num_samples = num_samples  # Number of samples to train on.
        # Path to the data txt file on disk.
        self.data_path = data_path
        if model_file:
            self.model_file = model_file
        else:
            self.model_file = data_path[:data_path.find(".")] + "model"
            # variable initialization for variables used across methods
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None
        self.model = None
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.encoder_model = None
        self.decoder_model = None
        self.reverse_target_char_index = None
        self.input_texts = None
        self.max_decoder_seq_length = None
        self.target_token_index = None
        self.decoder_target_data = None
        self.input_token_index = None

    def learnModel(self):
        self.buildModel()
        self.trainModel()

    def testModel(self):
        self.runInference()
        self.generateDecodedSentences()

    def prepareData(self):
        # Vectorize the data.
        self.input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text, _ = line.split("\t")
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"
            self.input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])

        print("Number of samples:", len(self.input_texts))
        print("Number of unique input tokens:", self.num_encoder_tokens)
        print("Number of unique output tokens:", self.num_decoder_tokens)
        print("Max sequence length for inputs:", max_encoder_seq_length)
        print("Max sequence length for outputs:", self.max_decoder_seq_length)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        self.encoder_input_data = np.zeros(
            (len(self.input_texts), max_encoder_seq_length,
             self.num_encoder_tokens), dtype="float32"
        )
        self.decoder_input_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length,
             self.num_decoder_tokens), dtype="float32"
        )
        self.decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length,
             self.num_decoder_tokens), dtype="float32"
        )

        for i, (input_text, target_text) in enumerate(
                zip(self.input_texts, target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[
                    i, t, self.input_token_index[char]] = 1.0
            self.encoder_input_data[i, t + 1:,
            self.input_token_index[" "]] = 1.0
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[
                    i, t, self.target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[
                        i, t - 1, self.target_token_index[char]] = 1.0
            self.decoder_input_data[i, t + 1:,
            self.target_token_index[" "]] = 1.0
            self.decoder_target_data[i, t:, self.target_token_index[" "]] = 1.0

    def buildModel(self):
        # Define an input sequence and process it.
        encoder_inputs = keras.Input(shape=(None, self.num_encoder_tokens))
        encoder = keras.layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.Input(shape=(None, self.num_decoder_tokens))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True,
                                         return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(self.num_decoder_tokens,
                                           activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = keras.Model([encoder_inputs, decoder_inputs],
                                 decoder_outputs)

    def trainModel(self):
        self.model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy",
            metrics=["accuracy"])
        self.model.fit(
            [self.encoder_input_data, self.decoder_input_data],
            self.decoder_target_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
        )
        # Save model
        self.model.save(self.model_file)

    def runInference(self):
        self.model = keras.models.load_model(self.model_file)
        encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[
            2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(self.latent_dim,),
                                            name="input_3")
        decoder_state_input_c = keras.Input(shape=(self.latent_dim,),
                                            name="input_4")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(
                    decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence

    def generateDecodedSentences(self):
        for seq_index in range(20):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = self.encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq)
            print("-")
            print("Input sentence:", self.input_texts[seq_index])
            print("Decoded sentence:", decoded_sentence)
