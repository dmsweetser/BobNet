import sys
import os
import argparse
from lib.bob import *
from lib.utilities import *
import shutil
from lib.vector_store import VectorStore
from lib.config_manager import *
import codecs
import time

class BobNet:
    
    def __init__(self):
        load_config()
        self.ingest_dir = get_config("ingest_dir")
        self.ingest_archive_dir = get_config("ingest_archive_dir")
        self.import_dir = get_config("import_dir")
        self.share_dir = get_config("share_dir")
        self.max_results = get_config("max_results")
        self.total_cores = get_config("total_cores")
        self.model_config = {
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
        self.test_mode = get_config("test_mode")

        if self.test_mode:
            self.archive_ingested_files = False
            self.generate_bob_for_sharing = True
            current_ticks = int(time.time())
            self.vector_store_file_name = f"{str(current_ticks)}.db"
            self.vector_store = VectorStore(vector_store_file_name)    
            sys.argv.append("What are blueberries?")
        else:
            self.archive_ingested_files = True
            self.generate_bob_for_sharing = True
            self.vector_store = VectorStore()
            
        self.make_dirs()
        self.ingest_training_data()
        self.import_bob_files()

    def make_dirs(self):
        os.makedirs(self.ingest_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ingest_dir, self.ingest_archive_dir), exist_ok=True)
        os.makedirs(self.import_dir, exist_ok=True)
        os.makedirs(self.share_dir, exist_ok=True)
        
    def ingest_training_data(self):
        if len(os.listdir(self.ingest_dir)) > 0:
            for file in os.listdir(self.ingest_dir):
                if "archive" in file:
                    continue
                full_file_path = os.path.join(self.ingest_dir, file)
                with codecs.open(full_file_path, 'rU', encoding='utf-8') as ingest_file:
                    training_text_raw = ingest_file.read()
                    process_text(
                        training_text_raw, 
                        self.model_config, 
                        self.generate_bob_for_sharing, 
                        self.share_dir, 
                        self.import_dir,
                        self.total_cores)
                if self.archive_ingested_files:
                    shutil.move(full_file_path, os.path.join(self.ingest_dir, self.ingest_archive_dir, file))
                   
    def ingest_single_training_text(self, training_data):
        if len(os.listdir(self.ingest_dir)) > 0:
            for file in os.listdir(self.ingest_dir):
                if "archive" in file:
                    continue
                full_file_path = os.path.join(self.ingest_dir, file)
                with codecs.open(full_file_path, 'rU', encoding='utf-8') as ingest_file:
                    training_text_raw = ingest_file.read()
                    process_text(
                        training_text_raw, 
                        self.model_config, 
                        self.generate_bob_for_sharing, 
                        self.share_dir, 
                        self.import_dir,
                        self.total_cores)
                    
    def import_bob_files(self):
        if len(os.listdir(self.import_dir)) > 0:
            for file in os.listdir(self.import_dir):
                if '.bob' not in file:
                    continue
                full_file_path = os.path.join(self.import_dir, file)
                with open(full_file_path, 'r') as import_file:
                    new_bob = Bob(import_file.read())
                    self.vector_store.add_vector(
                        new_bob.training_data,
                        new_bob.save_bob())
                os.remove(full_file_path)

    def infer(self, prompt):        
        output = prompt
        results = []
        probabilities = []
        previous_result = {}        
        while True:
            bob_net = []
            bob_data = vector_store.search(output, self.max_results)
            for entry in bob_data:
                bob = Bob()
                bob.load_bob(entry)
                bob_net.append(bob)
            
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

if __name__ == "__main__":
    
    bob_net = BobNet()
    if len(sys.argv) > 1:
        bob_net.infer(sys.argv[1])

      
        
            
            