import codecs
import multiprocessing
from multiprocessing import Pool
from functools import partial
from lib.bob import *
import re

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
            if start_index < len(words) and words[start_index] == string[start_index:start_index + 1]:
                word = words[start_index]
                if len(re.sub(r'\s+$', '', word)) + len(string[start_index:end_index]) >= chunk_size:
                    chunks.append(string[start_index:end_index])
                    total_words -= 1
                    start_index = end_index
                else:
                    start_index += len(word) + len(re.sub(r'\s+$', '', word))
            else:
                chunks.append(string[start_index:])
                start_index += len(string[start_index:start_index + 1])
                total_words -= 1
        else:
            chunks.append(string[start_index:])
            break

    return chunks

def process_training_text(training_text, config, generate_bob_for_sharing, share_dir, import_dir):
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

def process_file(full_file_path, config, generate_bob_for_sharing, share_dir,import_dir, total_cores, process_as_chunks = True):
    with codecs.open(full_file_path, 'rU', encoding='utf-8') as ingest_file:
        training_text_raw = ingest_file.read()
        if process_as_chunks and len(training_text_raw) > config["context_length"]:
            split_training_text = string_chunks(training_text_raw, config["context_length"])
            lengths = [len(s) for s in split_training_text]
            print(f"Models to be generated: {len(split_training_text)}")
            print(f"Total text length for all chunks: {sum(lengths)}")
            partial_process = partial(process_training_text, config=config,
                                    generate_bob_for_sharing=generate_bob_for_sharing, share_dir=share_dir, import_dir=import_dir)
            with Pool(processes=total_cores) as pool:
                pool.map(partial_process, split_training_text)
        else:
            process_training_text(training_text_raw, config, generate_bob_for_sharing, share_dir, import_dir)
            
def remove_duplicates(line):
    pattern = re.compile(r'(.*)(\s+)(\1+)$')
    result = re.sub(pattern, r'\1', line)
    return result