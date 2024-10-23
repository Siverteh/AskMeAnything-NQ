import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import re
from bs4 import BeautifulSoup
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class YesNoDataset(Dataset):
    def __init__(self, data, tokenizer_name='bert-base-uncased', max_length=512, balance=False):
        """
        Initializes the dataset by loading and preprocessing data.

        Args:
            data (list): List of data entries.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Maximum sequence length for the tokenizer.
            balance (bool): Whether to balance the dataset by oversampling.
        """
        print(f"Processing {len(data)} entries.")

        if balance:
            data = self.balance_data(data)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Preprocess and tokenize data upfront
        self.inputs = []
        self.labels = []

        for entry in data:
            question = entry['question_text']
            # Get context
            context = self.get_relevant_context(entry)
            context = self.clean_text(context)

            combined_input = f"{question} [SEP] {context}"

            #print(combined_input)
            #print("--------------------------------------------------------------------------")

            # Tokenize combined input
            inputs = self.tokenizer(
                combined_input,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.inputs.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)
            })

            label = self.vote_on_yes_no(entry['annotations'])
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def balance_data(self, data):
        """
        Balances the dataset by oversampling the minority class.

        Args:
            data (list): List of data entries.

        Returns:
            list: Balanced list of data entries.
        """
        from collections import Counter

        # Separate entries by label
        yes_entries = []
        no_entries = []

        for entry in data:
            label = self.vote_on_yes_no(entry['annotations'])
            if label == 1:
                yes_entries.append(entry)
            else:
                no_entries.append(entry)

        # Determine which class is the minority
        if len(yes_entries) > len(no_entries):
            majority_class = yes_entries
            minority_class = no_entries
        else:
            majority_class = no_entries
            minority_class = yes_entries

        # Calculate how many samples are needed to balance the classes
        difference = len(majority_class) - len(minority_class)

        # Oversample the minority class
        minority_oversampled = minority_class.copy()
        if difference > 0:
            oversampled_entries = random.choices(minority_class, k=difference)
            minority_oversampled.extend(oversampled_entries)

        # Combine the balanced classes
        balanced_data = majority_class + minority_oversampled
        random.shuffle(balanced_data)

        print(f"Balanced dataset with {len(yes_entries)} 'YES' and {len(no_entries)} 'NO' entries.")
        print(f"After balancing, each class has {len(minority_oversampled)} entries.")

        return balanced_data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'label': self.labels[idx]
        }

    def clean_text(self, text):
        """
        Cleans the input text by removing HTML tags and unnecessary whitespace.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        # Remove HTML tags using BeautifulSoup
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

        # Remove extra whitespace and lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()

        return text

    def get_relevant_context(self, entry, top_n=5):
        """
        Extracts the most relevant context for the question using TF-IDF similarity.

        Args:
            entry (dict): A single data entry containing the document and annotations.
            top_n (int): Number of top sentences to include as context.

        Returns:
            str: The extracted context text.
        """
        nltk.download('punkt', quiet=True)

        question = entry['question_text']
        document = entry['document_text']
        sentences = nltk.sent_tokenize(document)

        # If the document is too short, return it as is
        if len(sentences) <= top_n:
            return ' '.join(sentences)

        # Compute TF-IDF scores
        vectorizer = TfidfVectorizer().fit([question] + sentences)
        question_vec = vectorizer.transform([question])
        sentence_vecs = vectorizer.transform(sentences)

        # Compute cosine similarity
        similarities = cosine_similarity(question_vec, sentence_vecs).flatten()

        # Select top N sentences
        top_indices = similarities.argsort()[-top_n:]
        top_sentences = [sentences[i] for i in top_indices]

        # Combine top sentences as context
        context = ' '.join(top_sentences)
        return context

    def vote_on_yes_no(self, annotations):
        """
        Determines the label based on the majority vote of the annotations.

        Args:
            annotations (list): List of annotation dictionaries.

        Returns:
            int: 1 for 'YES', 0 for 'NO'.
        """
        yes_count = sum(1 for ann in annotations if ann['yes_no_answer'] == 'YES')
        no_count = sum(1 for ann in annotations if ann['yes_no_answer'] == 'NO')

        # If tie or no votes, default to 'NO'
        if yes_count > no_count:
            return 1  # 'YES'
        else:
            return 0  # 'NO'
