import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import json
from bs4 import BeautifulSoup
import re

class YesNoDataset(Dataset):
    def __init__(self, json_file, tokenizer_name='bert-base-uncased', max_length=512):
        print("Loading dataset...")
        with open(json_file, 'r') as f:
            self.data = [json.loads(line.strip()) for line in f if line.strip()]
        
        print(f"Loaded {len(self.data)} entries.")

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.cleaning_pattern = re.compile(r'<.*?>')  # Regex pattern to clean up HTML

    def clean_text(self, text):
        # Remove HTML tags using BeautifulSoup
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

        # Remove Wikipedia-specific phrases and meta information
        text = re.sub(r'jump\s*to\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(navigation|search|see also|this article is about|for other uses|edit)\b', '', text, flags=re.IGNORECASE)
        
        # Remove metadata-related keywords (refining to remove production details, names, etc.)
        text = re.sub(r'\b(executive producer|running time|company|release date|no\. of episodes|original release|website|production|chronology|external links)\b', '', text, flags=re.IGNORECASE)

        # Remove disambiguation phrases
        text = re.sub(r'for .*? see .*?\.', '', text, flags=re.IGNORECASE)

        # Remove parentheses and brackets with content inside
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)

        # Normalize punctuation and remove extra spaces
        text = re.sub(r'[\'`]+', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Split text into sentences
        sentences = text.split('.')

        # Filter and prioritize meaningful sentences
        cleaned_sentences = [s for s in sentences if len(s) > 5]

        # Keep only the first few meaningful sentences (2-3 sentences should suffice)
        cleaned_text = '. '.join(cleaned_sentences[:5])

        return text.strip().lower()



    def vote_on_yes_no(self, annotations):
        # Determine label based on majority vote
        yes_count = sum(1 for ann in annotations if ann['yes_no_answer'] == 'YES')
        no_count = sum(1 for ann in annotations if ann['yes_no_answer'] == 'NO')
        return 1 if yes_count > no_count else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        question = entry['question_text']
        document = self.clean_text(entry['document_text'])

        # Combine question and document
        combined_input = f"{question} [SEP] {document}"

        # Tokenize
        inputs = self.tokenizer(
            combined_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get label
        label = self.vote_on_yes_no(entry['annotations'])

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
