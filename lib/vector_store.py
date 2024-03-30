import sqlite3
import numpy as np
from gensim.models import KeyedVectors
from annoy import AnnoyIndex
import json
import nltk
nltk.download('punkt')

class VectorStore:
    def __init__(self):
        self.index_dim = 300
        self.db = sqlite3.connect('vector_store.db')
        self.create_table()
        # Download from https://github.com/mmihaltz/word2vec-GoogleNews-vectors
        self.word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',  limit=500000, binary=True) 
        self.index = self.init_index()

    def create_table(self):
        cursor = self.db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS vectors (
                            id INTEGER PRIMARY KEY,
                            vector BLOB,
                            raw_text TEXT,
                            model_data BLOB)''')

    def add_vector(self, raw_text, model_data):
        # Vectorize the text using Word2Vec
        words = nltk.word_tokenize(raw_text.lower())  # Tokenize the text
        vector = np.zeros((self.index_dim,), dtype=np.float32)  # Initialize vector with shape (300,)
        count = 0
        for word in words:
            if word in self.word2vec_model:
                vector += self.word2vec_model[word]
                count += 1
        if count > 0:
            vector /= count  # Average the word vectors
        else:
            return  # Skip if no word vectors found

        id = self.index.get_n_items()
        self.index.add_item(id, vector)
        cursor = self.db.cursor()

        # Convert model_data to bytes
        model_data_bytes = json.dumps(model_data).encode('utf-8')

        cursor.execute('''INSERT INTO vectors (id, vector, raw_text, model_data)
                        VALUES (?, ?, ?, ?)''', (id, vector.tobytes(), raw_text, sqlite3.Binary(model_data_bytes)))
        self.db.commit()
        self.index.drop
        self.index.build(10)  # Build the index after adding all items


    def search(self, query_text):
        # Vectorize the query text using Word2Vec
        words = nltk.word_tokenize(query_text.lower())
        query_vector = np.zeros(self.index_dim, dtype=np.float32)
        count = 0
        for word in words:
            if word in self.word2vec_model:
                query_vector += self.word2vec_model[word]
                count += 1
        if count > 0:
            query_vector /= count
        else:
            return []

        indices = self.index.get_nns_by_vector(query_vector, 10)
        results = []
        for i in indices:
            cursor = self.db.cursor()
            cursor.execute('''SELECT model_data FROM vectors WHERE id = ?''', (i,))
            row = cursor.fetchone()
            results.append(row[0])
        return results

    def has_records(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM vectors")
        count = cursor.fetchone()[0]
        return count > 0