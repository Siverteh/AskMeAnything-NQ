import torch
from torch.utils.data import Dataset

class LongAnswerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        if entry is None:
            print(f"Warning: Entry at index {idx} is None.")
            return None 

        input_ids = torch.tensor(entry['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(entry['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(entry['token_type_ids'], dtype=torch.long)
        label = torch.tensor(entry['label'], dtype=torch.float)  

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': label,  
            'question': entry.get('question', ''),
            'candidate_text': entry.get('candidate_text', ''),
        }