import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import re
from bs4 import BeautifulSoup
from transformers import LongformerTokenizer, RobertaTokenizer, DebertaTokenizer
import numpy as np

class YesNoDataset(Dataset):
    def __init__(self, data, tokenizer_name='bert-base-uncased', max_length=512):
        print(f"Processing {len(data)} entries.")

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        #self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        #self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

        self.max_length = max_length

        self.inputs = []
        self.labels = []

        for entry in data:
            question = entry['question_text']
            annotations = entry['annotations']
            document_text = entry['document_text']
            tokens = document_text.split()

            long_answer = annotations[0]['long_answer']
            start_token = long_answer['start_token']
            end_token = long_answer['end_token']

            if start_token != -1 and end_token != -1:
                answer_tokens = tokens[start_token:end_token]
                answer_text = ' '.join(answer_tokens)
                answer_text = self.clean_text(answer_text)

                #print("Question:", question)
                #print("-----------------------------------------------")
                #print("Answer:", answer_text)

                inputs = self.tokenizer(
                    question,
                    answer_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation='only_second',
                    return_tensors='pt'
                )

                self.inputs.append({
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0)
                })

                yes_no_answer = annotations[0]['yes_no_answer']
                label = 1 if yes_no_answer == 'YES' else 0
                self.labels.append(label)  

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def clean_text(self, text):
        """
        Cleans the input text by removing HTML tags and unnecessary whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

        text = re.sub(r'\s+', ' ', text).strip().lower()

        return text
