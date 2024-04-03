def string_chunks(string, chunk_size):
    step = chunk_size // 2
    return [string[i:i + chunk_size] for i in range(0, len(string) - chunk_size, step)]