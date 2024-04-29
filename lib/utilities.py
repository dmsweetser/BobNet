
from multiprocessing import Pool
from functools import partial
from lib.bob import *
import re
import time

def string_chunks(string, chunk_size):
    pattern = re.compile(r'\w+\s+')
    words = re.findall(pattern, string)
    total_words = len(words)
    step = round(chunk_size * 0.9)

    start_index = 0
    chunks = []
    while total_words > 0:
        end_index = min(start_index + chunk_size, len(string))
        if end_index > start_index:
            # Find the nearest space before the end index
            while end_index < len(string) and string[end_index] != ' ':
                end_index -= 1
            
            # If no space is found, move the end index to the end of the last word in the chunk
            if end_index == start_index:
                end_index = min(start_index + chunk_size, len(string))
                while end_index < len(string) and string[end_index] != ' ':
                    end_index += 1
            
            # Add the chunk if it's not empty
            if start_index != end_index:
                chunks.append(string[start_index:end_index])
                start_index += step  # Move the start index forward by the overlap amount
                total_words -= 1
            else:
                # If no space is found within the chunk size, move forward without overlap
                start_index += chunk_size
        else:
            chunks.append(string[start_index:])
            break

    return chunks

def process_training_text_chunk(training_text, config, generate_bob_for_sharing, share_dir, import_dir):
    try:
        new_bob = Bob(config=config, training_data=training_text)
        if generate_bob_for_sharing:
            current_ticks = int(time.time())
            file_name = f"{str(current_ticks)}.bob"
            share_full_path = os.path.join(share_dir, file_name)
            with open(share_full_path, "w") as share_file:
                share_file.write(json.dumps(new_bob.save_bob(), indent=4))
            import_full_path = os.path.join(import_dir, file_name)
            with open(import_full_path, "w") as import_file:
                import_file.write(json.dumps(new_bob.save_bob(), indent=4))
    except Exception as e:
        print(f"Exception encountered when processing training data:\n{training_text}\n\nException:\n{str(e)}")

def process_text(training_text_raw, config, generate_bob_for_sharing, share_dir,import_dir, total_cores, process_as_chunks = True):
        if process_as_chunks and len(training_text_raw) > config["context_length"]:
            split_training_text = string_chunks(training_text_raw, config["context_length"])
            lengths = [len(s) for s in split_training_text]
            print(f"Models to be generated: {len(split_training_text)}")
            print(f"Total text length for all chunks: {sum(lengths)}")
            partial_process = partial(process_training_text_chunk, config=config,
                                    generate_bob_for_sharing=generate_bob_for_sharing, share_dir=share_dir, import_dir=import_dir)
            with Pool(processes=total_cores) as pool:
                pool.map(partial_process, split_training_text)
        else:
            process_training_text_chunk(training_text_raw, config, generate_bob_for_sharing, share_dir, import_dir)
            
def remove_duplicates(line):
    pattern = re.compile(r'(.*)(\s+)(\1+)$')
    result = re.sub(pattern, r'\1', line)
    return result