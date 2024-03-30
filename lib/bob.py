import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Embedding, Bidirectional, GRU, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Add
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import datetime
import time
import string
import base64
import io
import tempfile

class Bob:
    def __init__(self, existing_model_path = "", config = "", training_data = "", model_dir=""):

        if existing_model_path != "":
            self._load_bob(existing_model_path)
        elif training_data != "" :
            self.end_token = '[e]'
            self.delimiter = '[m]'
            self.model_dir = model_dir
            self.populate_from_config(config)
            self.tokenizer = None
            self.model = None
            self._build_bob(training_data)
            self._save_bob(model_dir)
        
    def populate_from_config(self, config):
        self.config = config
        self.context_length = config["context_length"]
        self.embedding_dim = config["embedding_dim"]
        self.lstm_units = config["lstm_units"]
        self.hidden_dim = config["hidden_dim"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.dropout = config["dropout"]
        self.recurrent_dropout = config["recurrent_dropout"]
        self.temperature = config["temperature"]
        self.repetition_penalty = config["repetition_penalty"]        
        
    def infer(self, seed_text):
        try:
            model = self.model
            tokenizer = self.tokenizer
            sequence_length = self.context_length

            # Preprocess seed text
            seed_text = seed_text.translate(str.maketrans('', '', string.punctuation))
            generated_text = seed_text.lower()

            result = ""

            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")

            predicted_probs = model.predict(token_list, verbose=0)[0]

            # Find the index of the token with the highest probability
            max_prob_index = np.argmax(predicted_probs)

            # Get the word corresponding to the index
            result = tokenizer.index_word.get(max_prob_index, "")

            # Get the probability of the selected token
            selected_token_prob = predicted_probs[max_prob_index]
            return result, selected_token_prob
        except Exception as e:
            print(e)
            return "", 0
        
    def _create_model(self, context_length, vocab_size, embedding_dim, lstm_units, hidden_dim):

        # Calculate parameters for each layer
        embedding_params = vocab_size * embedding_dim
        lstm1_params = 4 * ((embedding_dim + lstm_units) * lstm_units)
        lstm2_params = 4 * ((lstm_units + lstm_units) * lstm_units)
        dense1_params = (lstm_units * hidden_dim) + hidden_dim
        dense2_params = (hidden_dim * vocab_size) + vocab_size
        output_dense_params = (hidden_dim * vocab_size) + vocab_size

        # Total parameters
        total_params = embedding_params + lstm1_params + lstm2_params + dense1_params + dense2_params + output_dense_params

        print("Total parameters:", total_params)

        inputs = Input(shape=(context_length,))
        pipeline = Embedding(vocab_size, embedding_dim)(inputs)        
        pipeline = LSTM(lstm_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(pipeline)
        pipeline = LSTM(lstm_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(pipeline)
        pipeline = Dense(hidden_dim, activation='relu')(pipeline)
        pipeline = Dense(vocab_size)(pipeline)  # No activation here

        # Apply softmax activation here
        outputs = Dense(vocab_size, activation='softmax')(pipeline)

        # Define the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with sparse categorical crossentropy loss and Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

    def _preprocess_data(self, text_data_arr):
        
        tokenizer = Tokenizer(lower=True, filters='')
        tokenizer.fit_on_texts(text_data_arr)
        sequences = tokenizer.texts_to_sequences(text_data_arr)

        vocab_size = len(tokenizer.word_index) + 1

        input_sequences = []
        output_sequences = []

        for sequence in sequences:
            for i in range(1, len(sequence)):
                input_sequence = sequence[:i]
                input_padding = pad_sequences([input_sequence], maxlen=self.context_length, padding="pre")[0]
                output_sequence = sequence[i]
                input_sequences.append(input_padding)
                output_sequences.append(output_sequence)

        self.tokenizer = tokenizer

        return np.array(input_sequences), np.array(output_sequences), vocab_size

    def _train_model(self, model, input_sequences, output_sequences, epochs, batch_size):
        model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

    def _save_bob(self, file_path):

        temp_file_path = tempfile.mktemp(suffix='.keras')
        with open(temp_file_path, 'wb') as temp_file:
            self.model.save(temp_file.name)
        with open(temp_file_path, 'rb') as temp_file:
            model_base64 = base64.b64encode(temp_file.read()).decode('utf-8')
            
        os.remove(temp_file_path)

        tokenizer_json = self.tokenizer.to_json()
        tokenizer_base64 = base64.b64encode(tokenizer_json.encode()).decode()

        data = {
            "model": model_base64,
            "tokenizer": tokenizer_base64,
            "config": self.config
        }
        
        current_ticks = int(time.time())
        file_name = f"{str(current_ticks)}.bob"
        full_path = os.path.join(self.model_dir, file_name)

        with open(full_path, "w") as file:
            file.write(json.dumps(data, indent=4))

    def _load_bob(self, file_path):
        
        json_data = ""
        
        if os.path.exists(file_path):
            json_data = open(file_path, "r").read()
        else:
            raise ValueError(f"File path {file_path} does not exist")
        
        data = json.loads(json_data)

        model_base64 = data["model"]
        tokenizer_base64 = data["tokenizer"]

        tokenizer_json = base64.b64decode(tokenizer_base64.encode()).decode()
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
             
        temp_file_path = tempfile.mktemp(suffix='.keras')
        with open(temp_file_path, 'wb') as temp_file:
            decoded_model = base64.b64decode(model_base64)
            temp_file.write(decoded_model)
        self.model = tf.keras.models.load_model(temp_file_path)
        os.remove(temp_file_path)
        
        self.populate_from_config(data["config"])
            
    def _build_bob(self, input_data):
        text_data_arr = []
        text_data_arr.append(f"{input_data}{self.end_token}")
        input_sequences, output_sequences, vocab_size = self._preprocess_data(text_data_arr)
        self.model = self._create_model(self.context_length, vocab_size, self.embedding_dim, self.lstm_units, self.hidden_dim)
        self._train_model(self.model, input_sequences, output_sequences, self.epochs, self.batch_size)