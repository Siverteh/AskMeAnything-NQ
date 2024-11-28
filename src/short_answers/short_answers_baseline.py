import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support

def load_data(file_path, subset_size=None):
    with open(file_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    if subset_size is not None and subset_size < len(data):
        data = random.sample(data, subset_size)
    return data

def prepare_lr_data(dataset):
    inputs = []
    targets = []

    for entry in dataset:
        question = entry['question_text']
        annotations = entry['annotations']
        document_text = entry['document_text']
        tokens = document_text.split()

        long_answer = annotations[0]['long_answer']
        long_start_token = long_answer['start_token']
        long_end_token = long_answer['end_token']

        if long_start_token != -1 and long_end_token != -1:
            long_answer_tokens = tokens[long_start_token:long_end_token]
            long_answer_text = ' '.join(long_answer_tokens)

            if annotations[0]['short_answers']:
                short_answer = annotations[0]['short_answers'][0]
                short_start_token = short_answer['start_token']
                short_end_token = short_answer['end_token']

                short_answer_tokens = tokens[short_start_token:short_end_token]
                short_answer_text = ' '.join(short_answer_tokens)

                inputs.append(f"{question} {long_answer_text}")
                targets.append(short_answer_text)

    return inputs, targets

def train_lr_baseline(train_inputs, train_targets, val_inputs, val_targets):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(train_inputs)
    X_val = vectorizer.transform(val_inputs)

    param_grid = {
        'C': [0.1, 1, 10],  
        'solver': ['liblinear', 'lbfgs'], 
        'max_iter': [100, 200, 500] 
    }
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid,
        scoring='f1_weighted',
        cv=3,
        verbose=2
    )
    grid_search.fit(X_train, train_targets)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    predictions = best_model.predict(X_val)

    precision, recall, f1, _ = precision_recall_fscore_support(val_targets, predictions, average='weighted')
    exact_match = sum([1 for pred, target in zip(predictions, val_targets) if pred == target]) / len(val_targets)

    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1: {f1:.4f}")
    print(f"Validation Exact Match (EM): {exact_match:.4f}")

    print("Classification Report:")
    print(classification_report(val_targets, predictions))

    return best_model, vectorizer

if __name__ == '__main__':
    train_file = 'short_answers_only-train.jsonl'
    val_file = 'short_answers_only-dev.jsonl'

    train_data = load_data(train_file, 25000)  
    val_data = load_data(val_file, 5000)

    train_inputs, train_targets = prepare_lr_data(train_data)
    val_inputs, val_targets = prepare_lr_data(val_data)

    model, vectorizer = train_lr_baseline(train_inputs, train_targets, val_inputs, val_targets)
