import sys
import os
import argparse
from lib.bob import *
from lib.utilities import *
import shutil
from lib.vector_store import VectorStore
from lib.config_manager import *
import codecs

# Load existing config or set defaults
load_config()

ingest_dir = get_config("ingest_dir")
ingest_archive_dir = get_config("ingest_archive_dir")
import_dir = get_config("import_dir")
share_dir = get_config("share_dir")
max_results = get_config("max_results")
total_cores = get_config("total_cores")

model_config = {
    "context_length": get_config("context_length"),
    "embedding_dim": get_config("embedding_dim"),
    "lstm_units": get_config("lstm_units"), 
    "hidden_dim": get_config("hidden_dim"),
    "epochs": get_config("epochs"),
    "batch_size": get_config("batch_size"),
    "learning_rate": get_config("learning_rate"),
    "dropout": get_config("dropout"),
    "recurrent_dropout": get_config("recurrent_dropout"),
    "temperature": get_config("temperature"),
    "repetition_penalty": get_config("repetition_penalty")
}

# Use this for doing clean repeated tests
test_mode = get_config("test_mode")

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
            process_file(full_file_path, model_config, generate_bob_for_sharing, share_dir, import_dir, total_cores)
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
        previous_result = {}
       
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
                
            if results[0] in previous_result:
                previous_result[results[0]] += 1
            else:
                previous_result[results[0]] = 1

            if previous_result[results[0]] >= 3:
                output = remove_duplicates(output)
                print("Duplicate result detected three times in a row. Exiting loop.")
                print(f"Final Output: {output}")
                break

            matching_result = results[0]

            output += " " + matching_result

            print(output)

            if "[e]" in output:
                break

            results = []
            probabilities = []