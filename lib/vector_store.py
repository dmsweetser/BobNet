import sqlite3
import numpy as np
from faiss import IndexFlatL2, IndexIVFInt32, FAISS_JDIFF_1000
from sklearn.feature_extraction.text import TfidfVectorizer

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
        return IndexFlatL2(self.index_dim)

    def add_vector(self, raw_text, model_data):
        vector = self.vectorizer.fit_transform([raw_text]).toarray().astype(np.float32)
        np_vector = np.asarray(vector, dtype=np.float32)
        self.index.add(np_vector)
        cursor = self.db.cursor()
        id = cursor.lastrowid
        self.index.add(np_vector)
        cursor.execute('''INSERT INTO vectors (id, vector, raw_text, model_data)
                          VALUES (?, ?, ?, ?)''', (id, np.tobytes(np_vector), raw_text, np.tobytes(model_data)))
        self.db.commit()

    def search(self, query_text):
        np_query_vector = self.vectorizer.fit_transform([query_text]).toarray().astype(np.float32)
        distances, indices = self.index.search(np_query_vector, FAISS_JDIFF_1000)
        results = []
        for i in indices:
            cursor = self.db.cursor()
            cursor.execute('''SELECT model_data FROM vectors WHERE id = ?''', (i[0],))
            model_data = cursor.fetchone()[0]
            results.append(model_data)
        return results[:10]
    
    def has_records(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM vectors")
        count = cursor.fetchone()[0]
        return count > 0