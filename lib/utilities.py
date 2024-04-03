import textwrap
import math

def wrap_text(text, width):
    lines = textwrap.wrap(text, width=width)
    offset = int(math.floor((width / 2) + 0.5))
    start_indices = [i for i in range(0, len(lines), width)]
    start_indices = [max(start_indices[i] + width + offset, start_indices[i]) for i in range(len(start_indices))]
    overlapped_lines = [lines[i][:start_indices[i] - offset] for i in range(len(lines))]
    return overlapped_lines