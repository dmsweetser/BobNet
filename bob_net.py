import sys
import os
import argparse
from lib.bob import *
import shutil
from lib.vector_store import VectorStore

ingest_dir = "ingest"
ingest_archive_dir = "ingest\\archive\\"
import_dir = "import"
share_dir = "share"

config = {
    "context_length": 256,
    "embedding_dim": 64,
    "lstm_units": 64, 
    "hidden_dim": 64,
    "epochs": 60,
    "batch_size": 64,
    "learning_rate": 0.015,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "temperature": 1.0,
    "repetition_penalty": 1.0
}

# Use this for doing clean repeated tests
test_mode = True

if test_mode:
    archive_ingested_files = False
    generate_bob_for_sharing = False
    current_ticks = int(time.time())
    vector_store_file_name = f"{str(current_ticks)}.db"
    vector_store = VectorStore(vector_store_file_name)    
    sys.argv.append("What is your name?")
else:
    archive_ingested_files = True
    generate_bob_for_sharing = True
    vector_store = VectorStore()

if __name__ == "__main__":
    
    os.makedirs(ingest_dir, exist_ok=True)
    os.makedirs(ingest_archive_dir, exist_ok=True)
    os.makedirs(import_dir, exist_ok=True)
    os.makedirs(share_dir, exist_ok=True)
    
    if len(os.listdir(ingest_dir)) > 0:
        for file in os.listdir(ingest_dir):
            if "archive" in file:
                continue
            full_file_path = os.path.join(ingest_dir, file)
            with open(full_file_path, 'r') as ingest_file:
                training_text = ingest_file.read()
                new_bob = Bob(config=config, training_data=training_text)
                vector_store.add_vector(
                    training_text,
                    new_bob.save_bob())
                
                if generate_bob_for_sharing:
                    current_ticks = int(time.time())
                    file_name = f"{str(current_ticks)}.bob"
                    full_path = os.path.join(share_dir, file_name)
                    with open(full_path, "w") as share_file:
                        share_file.write(json.dumps(new_bob.save_bob(), indent=4))
            if archive_ingested_files:
                shutil.move(full_file_path, os.path.join(ingest_archive_dir,file))
        
    if len(os.listdir(import_dir)) > 0:
        for file in os.listdir(import_dir):
            if '.bob' not in file:
                continue
            full_file_path = os.path.join(import_dir, file)
            with open(full_file_path, 'r') as import_file:
                new_bob = Bob(import_file)
                vector_store.add_vector(
                    new_bob.training_data,
                    new_bob.save_bob())

    if len(sys.argv) > 1:
        output = sys.argv[1]
        results = []
        probabilities = []
        
        bob_net = []
        
        bob_data = vector_store.search(output)
        
        for entry in bob_data:
            bob = Bob()
            bob.load_bob(entry)
            bob_net.append(bob)
        
        while True:
            
            for bob in bob_net:                
                result, probability = bob.infer(output)
                results.append(result)
                probabilities.append(probability)
                
            max_probability_index = probabilities.index(max(probabilities))
            matching_result = results[max_probability_index]    
            
            output += " " + matching_result
            
            print(output)
            
            if "[e]" in output:
                break
            
            results = []
            probabilities = []
    
    