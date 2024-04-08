import sys
import os
import argparse
from lib.bob import *
from lib.utilities import *
import shutil
from lib.vector_store import VectorStore
import codecs

ingest_dir = "ingest"
ingest_archive_dir = "ingest\\archive\\"
import_dir = "import"
share_dir = "share"
max_results = 3

config = {
    "context_length": 2048,
    "embedding_dim": 4096,
    "lstm_units": 64, 
    "hidden_dim": 4096,
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.015,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "temperature": 1.0,
    "repetition_penalty": 1.0
}

# Use this for doing clean repeated tests
test_mode = False

if test_mode:
    archive_ingested_files = False
    generate_bob_for_sharing = True
    current_ticks = int(time.time())
    vector_store_file_name = f"{str(current_ticks)}.db"
    vector_store = VectorStore(vector_store_file_name)    
    sys.argv.append("What are blueberries?")
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
            process_file(full_file_path, config, generate_bob_for_sharing, share_dir, import_dir)
            if archive_ingested_files:
                shutil.move(full_file_path, os.path.join(ingest_archive_dir,file))
        
    if len(os.listdir(import_dir)) > 0:
        for file in os.listdir(import_dir):
            if '.bob' not in file:
                continue
            full_file_path = os.path.join(import_dir, file)
            with open(full_file_path, 'r') as import_file:
                new_bob = Bob(import_file.read())
                vector_store.add_vector(
                    new_bob.training_data,
                    new_bob.save_bob())
            os.remove(full_file_path)

    if len(sys.argv) > 1:
        output = sys.argv[1]
        results = []
        probabilities = []
        previous_result = ""
        
        bob_net = []
        
        bob_data = vector_store.search(output, max_results)
        
        for entry in bob_data:
            bob = Bob()
            bob.load_bob(entry)
            bob_net.append(bob)
        
        while True:
            for bob in bob_net:                
                result, probability = bob.infer(output)
                results.append(result)
                probabilities.append(probability)

                if result == previous_result:
                    duplicate_count += 1
                else:
                    duplicate_count = 0

                if duplicate_count >= 3:
                    print("Duplicate result detected three times in a row. Exiting loop.")
                    output = output[:-len(previous_result)]
                    break

                previous_result = result

            matching_result = results[0]

            output += " " + matching_result

            print(output)

            if "[e]" in output:
                break

            results = []
            probabilities = []