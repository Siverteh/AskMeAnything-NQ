import torch
from torch.utils.data import Dataset
from transformers import DebertaTokenizerFast, RobertaTokenizerFast, BertTokenizerFast
import re
from bs4 import BeautifulSoup

class ShortAnswerDataset(Dataset):
    def __init__(self, data, tokenizer_name='microsoft/deberta-base', max_length=512, max_answers=3):
        print(f"Processing {len(data)} entries.")

        #self.tokenizer = DebertaTokenizerFast.from_pretrained(tokenizer_name)
        #self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.max_length = max_length
        self.max_answers = max_answers  

        self.inputs = []
        self.start_positions = []
        self.end_positions = []
        self.example_ids = []

        self.debug = False

        for entry_idx, entry in enumerate(data):
            question = entry['question_text']
            example_id = entry['example_id']
            annotations = entry['annotations']
            document_text = entry['document_text']
            tokens = document_text.split()

            long_answer = annotations[0]['long_answer']
            long_start_token = long_answer['start_token']
            long_end_token = long_answer['end_token']

            if long_start_token != -1 and long_end_token != -1:
                long_answer_tokens = tokens[long_start_token:long_end_token]
                long_answer_text = ' '.join(long_answer_tokens)
                long_answer_text = self.clean_text(long_answer_text)

                short_answers = annotations[0]['short_answers'][:self.max_answers]

                for short_answer in short_answers:
                    short_start_token = short_answer['start_token']
                    short_end_token = short_answer['end_token']

                    relative_short_start = short_start_token - long_start_token
                    relative_short_end = short_end_token - long_start_token

                    short_answer_tokens = long_answer_tokens[relative_short_start:relative_short_end]
                    short_answer_text = ' '.join(short_answer_tokens)
                    short_answer_text = self.clean_text(short_answer_text)

                    # Tokenize the question and context
                    inputs = self.tokenizer(
                        question,
                        long_answer_text,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation='only_second',
                        return_offsets_mapping=True,
                        return_token_type_ids=True,
                        return_tensors='pt'
                    )

                    offset_mapping = inputs['offset_mapping'][0]
                    sequence_ids = inputs.sequence_ids(0)

                    cleaned_long_answer_text = self.clean_text(long_answer_text)
                    cleaned_short_answer_text = self.clean_text(short_answer_text)

                    start_char_in_context = cleaned_long_answer_text.find(cleaned_short_answer_text)
                    if start_char_in_context == -1:
                        if self.debug:
                            print("Could not find short answer in the context, skipping example.")
                        continue  

                    end_char_in_context = start_char_in_context + len(cleaned_short_answer_text)

                    start_token_index = None
                    end_token_index = None

                    for idx, (offset, seq_id) in enumerate(zip(offset_mapping, sequence_ids)):
                        if seq_id != 1:
                            continue  
                        token_start_char, token_end_char = offset

                        if token_start_char is None or token_end_char is None:
                            continue  

                        if token_start_char >= self.max_length:
                            break  

                        if token_start_char <= start_char_in_context < token_end_char:
                            start_token_index = idx
                        if token_start_char < end_char_in_context <= token_end_char:
                            end_token_index = idx

                        if start_token_index is not None and end_token_index is not None:
                            break  

                    if start_token_index is None or end_token_index is None:
                        if self.debug:
                            print("Could not find token indices for the answer span, skipping example.")
                        continue  

                    if start_token_index >= self.max_length or end_token_index >= self.max_length:
                        if self.debug:
                            print(f"Skipping example with out-of-bounds positions: start {start_token_index}, end {end_token_index}")
                        continue  

                    self.inputs.append({
                        'input_ids': inputs['input_ids'].squeeze(0),
                        'attention_mask': inputs['attention_mask'].squeeze(0),
                        'token_type_ids': inputs['token_type_ids'].squeeze(0)
                    })
                    self.start_positions.append(start_token_index)
                    self.end_positions.append(end_token_index)
                    self.example_ids.append(example_id)

                    if self.debug:
                        print(f"Example ID: {example_id}")
                        print(f"Question: {question}")
                        print(f"Context: {cleaned_long_answer_text}")
                        print(f"Short Answer: {cleaned_short_answer_text}")
                        print(f"Start Token Index: {start_token_index}")
                        print(f"End Token Index: {end_token_index}")
                        print("-" * 50)
            else:
                if self.debug:
                    print(f"No valid long answer in example {example_id}, skipping.")
                continue  

        self.start_positions = torch.tensor(self.start_positions, dtype=torch.long)
        self.end_positions = torch.tensor(self.end_positions, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'token_type_ids': self.inputs[idx]['token_type_ids'],
            'start_positions': self.start_positions[idx],
            'end_positions': self.end_positions[idx],
            'example_id': self.example_ids[idx]
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

        text = re.sub(r'\s+', ' ', text).strip()

        return text
