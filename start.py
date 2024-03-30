import sys
import os
import argparse
from lib.bob import *

bob_net = []
model_dir = "models"
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

def parse_arguments(file_path=None):
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument("--file", type=str, default="", help="Path to input file.", required=False)

    args = parser.parse_args()

    if args.file:
        file_path = args.file
        print(f"Command-line argument: {file_path}")
        with open(file_path, 'r') as file:
            file_content = file.read()
            bob = Bob(config=config, training_data=file_content, model_dir=model_dir)          

if __name__ == "__main__":
    
    os.makedirs(model_dir, exist_ok=True)
                
    parse_arguments()
    
    if len(os.listdir(model_dir)) == 0:
        bob_net.append(Bob(config=config, training_data="What is your name? My name is Bob.", model_dir=model_dir))
        # bob_net.append(Bob(config=config, training_data="What is 2 + 2? 2 + 2 = 4.", model_dir=model_dir))
        # bob_net.append(Bob(config=config, training_data="What is 2 + 4? 2 + 4 = 6.", model_dir=model_dir))
    else:    
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                bob_net.append(Bob(file_path))


    output = "What is your name?"
    
    results = []
    probabilities = []
    
    # TODO use vector store to retrieve only required Bobs from BobNet based on the inquiry
    
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