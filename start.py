import sys
import os
import argparse
from lib.bob import *
from lib.vector_store import VectorStore

ingest_dir = "ingest"
import_dir = "import"
share_dir = "share"

config = {
    "context_length": 64,
    "embedding_dim": 64,
    "lstm_units": 64, 
    "hidden_dim": 64,
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.015,
    "dropout": 0.2,
    "recurrent_dropout": 0.2,
    "temperature": 1.0,
    "repetition_penalty": 1.0
}

vector_store = VectorStore(64)

if __name__ == "__main__":
    
    os.makedirs(ingest_dir, exist_ok=True)
    os.makedirs(import_dir, exist_ok=True)
    os.makedirs(share_dir, exist_ok=True)
    
    if len(os.listdir(ingest_dir)) > 0:
        for file in os.listdir(ingest_dir):
            with open(file, 'r') as ingest_file:
                training_text = ingest_file.read()
                vector_store.add_vector(
                    training_text,
                    Bob(config=config, training_data=training_text, model_dir=model_dir).save_bob())
        
    if len(os.listdir(import_dir)) > 0:
        for file in os.listdir(import_dir):
            with open(file, 'r') as import_file:
                new_bob = Bob(import_file)
                vector_store.add_vector(
                    new_bob.training_data,
                    new_bob.save_bob())
                
                current_ticks = int(time.time())
                file_name = f"{str(current_ticks)}.bob"
                full_path = os.path.join(share_dir, file_name)
                with open(full_path, "w") as share_file:
                    file.write(json.dumps(new_bob.save_bob(), indent=4))

    if len(sys.argv) > 1:
        output = sys.argv[1]
        results = []
        probabilities = []
        
        bob_net = vector_store.search(output)
        
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
    
    