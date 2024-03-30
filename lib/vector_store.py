import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import json

class VectorStore:
    def __init__(self, index_dim):
        self.index_dim = index_dim
        self.db = sqlite3.connect('vector_store.db')
        self.create_table()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.index = self.init_index()

    def create_table(self):
        cursor = self.db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS vectors (
                            id INTEGER PRIMARY KEY,
                            vector BLOB,
                            raw_text TEXT,
                            model_data BLOB)''')

    def init_index(self):
        self.index = AnnoyIndex(self.index_dim, 'angular')
        return self.index

    def add_vector(self, raw_text, model_data):
        vector = self.vectorizer.fit_transform([raw_text]).toarray().astype(np.float32)
        np_vector = np.asarray(vector, dtype=np.float32)
        # Ensure consistent length by padding or truncating
        if len(np_vector[0]) != self.index_dim:
            np_vector = self._resize_vector(np_vector)
        id = self.index.get_n_items()
        self.index.add_item(id, np_vector[0])  # Annoy expects 1D array
        cursor = self.db.cursor()

        # Convert model_data to bytes
        model_data_bytes = json.dumps(model_data).encode('utf-8')

        cursor.execute('''INSERT INTO vectors (id, vector, raw_text, model_data)
                        VALUES (?, ?, ?, ?)''', (id, np_vector.tobytes(), raw_text, sqlite3.Binary(model_data_bytes)))
        self.db.commit()
        self.index.build(10)  # Build the index after adding all items

    def _resize_vector(self, vector):
        if len(vector[0]) > self.index_dim:
            return vector[:, :self.index_dim]  # Truncate
        else:
            return np.pad(vector, ((0, 0), (0, self.index_dim - len(vector[0]))), mode='constant')  # Pad


    def search(self, query_text):
        np_query_vector = self.vectorizer.fit_transform([query_text]).toarray().astype(np.float32)
        indices = self.index.search(np_query_vector, 10)
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