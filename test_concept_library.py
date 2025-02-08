import pytest
from concept_library import ConceptLibrary, SQLiteStorage
import os
import numpy as np

# Fixture to create and tear down a test database
@pytest.fixture
def test_library():
    db_path = 'test_concepts.db'
    library = ConceptLibrary(storage_backend=SQLiteStorage(db_path=db_path))
    yield library  # Provide the library instance to the tests
    library.storage.close()
    os.remove(db_path)  # Clean up the database file
    if os.path.exists(db_path + ".annoy"):
        os.remove(db_path + ".annoy") #Clean up the annoy file

def test_add_and_retrieve_interaction(test_library):
    metadata = {"timestamp": "2024-10-27T11:00:00Z", "category": "test"}
    record_id = test_library.add_interaction("This is a test interaction.", metadata)
    assert record_id is not None

    results = test_library.search("test interaction", top_k=1)
    assert len(results) == 1
    assert results[0]['raw_text'] == "This is a test interaction."
    assert results[0]['metadata'] == metadata

def test_search_similarity(test_library):
    test_library.add_interaction("apple", {"type": "fruit"})
    test_library.add_interaction("banana", {"type": "fruit"})
    test_library.add_interaction("car", {"type": "vehicle"})

    results = test_library.search("orange", top_k=2)
    # We expect "apple" and "banana" to be more similar to "orange" than "car" is.
    assert len(results) == 2
     # Check if the results are in the expected order (most similar first)
    # This is a basic check; a more robust test would compare distances.
    assert results[0]['raw_text'] in ("apple", "banana")
    assert results[1]['raw_text'] in ("apple", "banana")
    assert results[0]['raw_text'] != results[1]['raw_text']  # Ensure they are different


def test_empty_search(test_library):
    results = test_library.search("something not present", top_k=5)
    assert len(results) == 0  # Should return an empty list if nothing is found

def test_annoy_index_creation_and_loading(tmpdir):
    # Use a temporary directory for this test
    db_path = str(tmpdir.join('test_concepts.db'))
    storage = SQLiteStorage(db_path=db_path)
    storage.close()

    # Check if the Annoy index file was created
    assert os.path.exists(db_path + '.annoy')

    # Create a new storage instance (should load the index)
    storage2 = SQLiteStorage(db_path=db_path)
    storage2.close()
    # If loading was successful, no error should occur.
