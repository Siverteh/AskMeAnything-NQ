import json

def display_short_answer_entry(input_file, entry_index=0):
    """
    This function reads a JSONL file and displays the specified entry that contains non-empty short answers.
    
    Args:
    input_file (str): Path to the input JSONL file.
    entry_index (int): Index of the entry with non-empty short answers to display.
    """
    filtered_entries = []
    
    with open(input_file, 'r') as infile:
        for line in infile:
            entry = json.loads(line)

            for annotation in entry['annotations']:
                if annotation.get('short_answers') and len(annotation['short_answers']) > 0:
                    filtered_entries.append(entry)
                    break  

    if entry_index < len(filtered_entries):
        print(f"Entry at index {entry_index} with non-empty short answers:")
        print(json.dumps(filtered_entries[entry_index], indent=4))
    else:
        print(f"No entry found at index {entry_index} with non-empty short answers.")

input_file = 'datasets/short_answers_only-dev.jsonl'  

entry_index = 1  

display_short_answer_entry(input_file, entry_index)
