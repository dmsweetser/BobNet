import codecs
from multiprocessing import Pool
from functools import partial
from lib.bob import *

def string_chunks(string, chunk_size):
    step = chunk_size * 0.75
    return [string[i:i + chunk_size] for i in range(0, len(string) - chunk_size, step)]

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

def process_file(full_file_path, config, generate_bob_for_sharing, share_dir,import_dir):
    with codecs.open(full_file_path, 'rU', encoding='utf-8') as ingest_file:
        training_text_raw = ingest_file.read()
        split_training_text = string_chunks(training_text_raw, config["context_length"] * 50)
        partial_process = partial(process_training_text, config=config,
                                   generate_bob_for_sharing=generate_bob_for_sharing, share_dir=share_dir, import_dir=import_dir)
        with Pool() as pool:
            pool.map(partial_process, split_training_text)