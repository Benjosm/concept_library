import sqlite3
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

class ConceptLibrary:
    def __init__(self, encoder_model_name='sentence-transformers/all-MiniLM-L6-v2', storage_backend=None):
        self.encoder = SentenceTransformer(encoder_model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        if storage_backend is None:
            self.storage = SQLiteStorage(embedding_dim=self.embedding_dim)
        else:
            self.storage = storage_backend

    def add_interaction(self, user_input, metadata):
        concept_vector = self.encoder.encode(user_input)
        record = {
            'vector': concept_vector.tolist(),
            'metadata': metadata,
            'raw_text': user_input
        }
        record_id = self.storage.store(record)
        self.storage.update_annoy_index(record_id, concept_vector)
        return record_id

    def search(self, query, top_k=10):
        query_vector = self.encoder.encode(query)
        return self.storage.search(query_vector, top_k)

class SQLiteStorage:
    def __init__(self, db_path='concepts.db', embedding_dim=384, annoy_metric='angular', n_trees=10):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.embedding_dim = embedding_dim
        self.annoy_index = AnnoyIndex(embedding_dim, annoy_metric)
        self.annoy_index_path = db_path + '.annoy'
        self.n_trees = n_trees
        self._create_table()
        self._load_annoy_index()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector TEXT,
                metadata TEXT,
                raw_text TEXT
            )
        ''')
        self.conn.commit()

    def _load_annoy_index(self):
        """Loads the Annoy index from disk, or builds it if it doesn't exist."""
        try:
            if os.path.exists(self.annoy_index_path):
                self.annoy_index.load(self.annoy_index_path)
                print(f"Loaded Annoy index from {self.annoy_index_path}")
            else:
                raise FileNotFoundError
        except (OSError, FileNotFoundError):
            print("Annoy index not found, building a new one...")
            self._build_annoy_index()

    def _build_annoy_index(self):
        """Builds the Annoy index from scratch"""
        self.annoy_index.unload()
        self.annoy_index = AnnoyIndex(self.embedding_dim, 'angular')
        
        self.cursor.execute('SELECT id, vector FROM concepts')
        for row in self.cursor:
            record_id, vector_str = row
            vector = np.array(json.loads(vector_str))
            self.annoy_index.add_item(record_id, vector)
            
        self.annoy_index.build(self.n_trees)
        self.annoy_index.save(self.annoy_index_path)
        print(f"Built new Annoy index at {self.annoy_index_path}")

    def update_annoy_index(self, record_id, vector):
        """Triggers a full rebuild of the index"""
        self._build_annoy_index()

    def store(self, record):
        self.cursor.execute('''
            INSERT INTO concepts (vector, metadata, raw_text)
            VALUES (?, ?, ?)
        ''', (json.dumps(record['vector']), json.dumps(record['metadata']), record['raw_text']))
        self.conn.commit()
        return self.cursor.lastrowid

    def search(self, query_vector, top_k):
        neighbor_ids = self.annoy_index.get_nns_by_vector(query_vector, top_k)
        results = []
        for record_id in neighbor_ids:
            self.cursor.execute('SELECT * FROM concepts WHERE id = ?', (record_id,))
            row = self.cursor.fetchone()
            if row:
                results.append({
                    'id': row[0],
                    'vector': json.loads(row[1]),
                    'metadata': json.loads(row[2]),
                    'raw_text': row[3]
                })
        return results

    def close(self):
        self.conn.close()

if __name__ == '__main__':
    # Test with clean database
    if os.path.exists('concepts.db'):
        os.remove('concepts.db')
    if os.path.exists('concepts.db.annoy'):
        os.remove('concepts.db.annoy')
        
    library = ConceptLibrary()
    
    library.add_interaction("How do I manage stress?", {"category": "health"})
    library.add_interaction("Python programming tips", {"category": "coding"})
    
    results = library.search("anxiety relief")
    print("Search Results:")
    for res in results:
        print(f"- {res['raw_text']} (Category: {res['metadata']['category']})")
    
    library.storage.close()