import json
import re
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
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

def get_relevant_context(entry):
    """
    Extracts the most relevant context for the question using TF-IDF similarity.

    Args:
        entry (dict): A single data entry containing the document and annotations.

    Returns:
        str: The extracted context text.
    """
    question = entry['question_text']
    document = entry['document_text']
    sentences = nltk.sent_tokenize(document)

    # If the document is too short, return it as is
    if len(sentences) <= 3:
        context = ' '.join(sentences)
        return context

    # Compute TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english').fit([question] + sentences)
    question_vec = vectorizer.transform([question])
    sentence_vecs = vectorizer.transform(sentences)

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(question_vec, sentence_vecs).flatten()

    # Select top N sentences
    top_n = 3
    top_indices = similarities.argsort()[-top_n:]
    top_sentences = [sentences[i] for i in top_indices]

    # Remove duplicates
    top_sentences = list(dict.fromkeys(top_sentences))

    # Combine top sentences as context
    context = ' '.join(top_sentences)
    return context

def vote_on_yes_no(annotations):
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

def load_and_preprocess_data(data_file):
    """
    Loads data from a JSONL file and preprocesses it.

    Args:
        data_file (str): Path to the JSONL data file.

    Returns:
        DataFrame: A pandas DataFrame containing preprocessed data.
    """
    # Load data
    with open(data_file, 'r') as f:
        raw_data = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(raw_data)} entries.")

    # Preprocess data
    data = []
    for entry in raw_data:
        question = entry['question_text']
        context = get_relevant_context(entry)
        context = clean_text(context)
        combined_text = f"{question} {context}"

        label = vote_on_yes_no(entry['annotations'])

        data.append({
            'text': combined_text,
            'label': label
        })

    df = pd.DataFrame(data)
    return df

def train_and_evaluate(df):
    """
    Trains a logistic regression model and evaluates it.

    Args:
        df (DataFrame): The preprocessed data.
    """
    # Split data into features and labels
    X = df['text']
    y = df['label']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['NO', 'YES']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NO', 'YES'])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Cross-validation scores
    cv_scores = cross_val_score(model, vectorizer.transform(X), y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

if __name__ == "__main__":
    data_file = 'simplified-yes-no-train.jsonl'
    df = load_and_preprocess_data(data_file)
    train_and_evaluate(df)
