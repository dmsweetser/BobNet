import sqlite3
import numpy as np
from gensim.models import KeyedVectors
import json
import nltk
import zlib

class VectorStore:
    def __init__(self, file_name = 'vector_store.db'):
        self.index_dim = 300
        self.db = sqlite3.connect(file_name)
        self.create_table()
        # Download from https://github.com/mmihaltz/word2vec-GoogleNews-vectors
        self.word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',  limit=500000, binary=True) 

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

        cursor = self.db.cursor()

        # Compress model_data using zlib
        compressed_model_data = zlib.compress(json.dumps(model_data).encode('utf-8'))

        # Insert into database with compressed model_data
        cursor.execute('''INSERT INTO vectors (vector, raw_text, model_data)
                        VALUES (?, ?, ?)''', (vector.tobytes(), raw_text, sqlite3.Binary(compressed_model_data)))
        self.db.commit()

    def search(self, query_text, max_results=10):
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

        cursor = self.db.cursor()
        cursor.execute('''SELECT id, vector FROM vectors''')
        rows = cursor.fetchall()

        # Calculate cosine similarity with each vector
        similarities = []
        for row in rows:
            vector_bytes = row[1]
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((row[0], similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Retrieve and decompress model_data for top results
        results = []
        for i in range(min(max_results, len(similarities))):
            vector_id = similarities[i][0]
            cursor.execute('''SELECT model_data FROM vectors WHERE id = ?''', (vector_id,))
            row = cursor.fetchone()
            decompressed_model_data = json.loads(zlib.decompress(row[0]))
            results.append(decompressed_model_data)

        return results

    def has_records(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM vectors")
        count = cursor.fetchone()[0]
        return count > 0
