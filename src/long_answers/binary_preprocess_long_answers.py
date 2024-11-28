import json
import pickle
import re
from bs4 import BeautifulSoup
from transformers import DebertaTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
from tqdm import tqdm
import random
from multiprocessing import Pool, Manager


def clean_html(html_text):
    """
    Cleans HTML tags from text and refines spacing to avoid issues with punctuation.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    clean_text = soup.get_text(separator=" ")

    clean_text = re.sub(r'\s+', ' ', clean_text)

    clean_text = re.sub(r'\s+([.,!?;:%\)])', r'\1', clean_text)

    clean_text = re.sub(r'([(\[{])\s+', r'\1', clean_text)

    clean_text = clean_text.strip()

    return clean_text



def is_informative_candidate(candidate_text):
    """
    Determines whether a candidate is informative.

    Args:
        candidate_text (str): The candidate text to evaluate.

    Returns:
        bool: True if the candidate is deemed informative, False otherwise.
    """
    num_tokens = len(candidate_text.split())
    if num_tokens < 25 or num_tokens > 512:
        return False 

    if re.search(
        r"(this article|learn how and when to remove|template message|multiple issues|unsourced material|original research)",
        candidate_text,
        re.IGNORECASE
    ):
        return False

    if candidate_text.lower().count("please help improve") > 1:
        return False

    if re.search(r"(wikipedia|free encyclopedia|editors|citation needed|disambiguation)", candidate_text, re.IGNORECASE):
        return False

    if re.search(
        r"(jump to navigation|jump to search|navigation menu|retrieved from|last edited)", 
        candidate_text, 
        re.IGNORECASE
    ):
        return False

    return True


def process_example(args):
    """
    Processes a single example for binary classification on long answer candidates.
    """
    example, tokenizer, max_seq_length = args
    instances = []

    question = example['question_text']
    document_text = example['document_text']
    annotations = example['annotations']
    long_answer_candidates = example['long_answer_candidates']

    top_level_candidates = [
        candidate for candidate in long_answer_candidates if candidate['top_level']
    ]

    annotated_long_answers = set()
    for annotation in annotations:
        la = annotation['long_answer']
        if la['start_token'] != -1 and la['end_token'] != -1:
            annotated_long_answers.add((la['start_token'], la['end_token']))

    if not annotated_long_answers:
        return instances  

    positive_candidate = None
    negative_candidates = []

    for i, candidate in enumerate(top_level_candidates):
        cand_start_token = candidate['start_token']
        cand_end_token = candidate['end_token']

        if cand_end_token > len(document_text.split()):
            continue

        raw_candidate_text = ' '.join(document_text.split()[cand_start_token:cand_end_token])
        candidate_text = clean_html(raw_candidate_text)

        if candidate_text.strip() == '' or not is_informative_candidate(candidate_text):
            continue 

        is_positive = any(
            cand_start_token <= la_start_token and cand_end_token >= la_end_token
            for la_start_token, la_end_token in annotated_long_answers
        )

        if is_positive and positive_candidate is None:
            positive_candidate = candidate_text
        elif not is_positive:
            if len(negative_candidates) < 3 and (not negative_candidates or i - negative_candidates[-1]['index'] > 2):
                negative_candidates.append({'index': i, 'text': candidate_text})

        if positive_candidate and len(negative_candidates) == 3:
            break

    if not positive_candidate or len(negative_candidates) < 3:
        return instances

    encoding = tokenizer.encode_plus(
        question,
        positive_candidate,
        truncation='longest_first',
        max_length=max_seq_length,
        padding='max_length',
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    instances.append({
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding['token_type_ids'],
        'label': 1,
        'question': question,
        'candidate_text': positive_candidate,
    })

    for negative in negative_candidates:
        encoding = tokenizer.encode_plus(
            question,
            negative['text'],
            truncation='longest_first',
            max_length=max_seq_length,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        instances.append({
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            'label': 0,
            'question': question,
            'candidate_text': negative['text'],
        })

    return instances



def preprocess_nq_data_binary_classification(data, tokenizer, max_seq_length=512, num_workers=8, target_examples=5000):
    """
    Preprocesses the NQ dataset using parallel processing, stopping once the target number of examples is reached.
    """
    processed_instances = []
    example_count = 0  

    with Pool(num_workers) as pool:
        args = [(example, tokenizer, max_seq_length) for example in data]
        for result in tqdm(pool.imap(process_example, args), total=len(data), desc="Processing examples"):
            if len(result) >= 4:  
                processed_instances.extend(result)
                example_count += 1

                if example_count >= target_examples:
                    print(f"Reached target of {target_examples} examples ({example_count * 4} instances).")
                    return processed_instances

    print(f"Total processed examples: {example_count}")
    print(f"Total processed instances: {len(processed_instances)}")
    return processed_instances


def load_nq_data(file_path, max_samples=None):
    """
    Load the NQ dataset from a JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            example = json.loads(line)
            data.append(example)
    return data


def main():
    train_file = 'simplified-train.jsonl'
    dev_file = 'simplified-dev.jsonl'

    print("Loading data...")
    train_data = load_nq_data(train_file, max_samples=100000) 
    dev_data = load_nq_data(dev_file, max_samples=10000)

    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    #tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')

    print("Preprocessing training data...")
    processed_train_data = preprocess_nq_data_binary_classification(
        train_data, tokenizer, num_workers=50, target_examples=25000
    )

    print("Preprocessing dev data...")
    processed_dev_data = preprocess_nq_data_binary_classification(
        dev_data, tokenizer, num_workers=50, target_examples=2500
    )

    with open('deberta_train_data_binary.pkl', 'wb') as f:
        pickle.dump(processed_train_data, f)
    with open('deberta_dev_data_binary.pkl', 'wb') as f:
        pickle.dump(processed_dev_data, f)

    print("Preprocessing complete.")
    print(f"Total training instances: {len(processed_train_data)}")
    print(f"Total dev instances: {len(processed_dev_data)}")



if __name__ == '__main__':
    main()
