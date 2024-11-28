import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, DebertaForSequenceClassification, DebertaTokenizerFast, RobertaForSequenceClassification, RobertaTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import nltk

nltk.download('punkt')

MAX_SEQ_LENGTH = 512        
MAX_SENTENCES = 5           
STRIDE = 3                  

class InferenceDataset(Dataset):
    """
    Custom Dataset for handling (question, span) pairs.
    """
    def __init__(self, encodings, span_texts):
        self.encodings = encodings
        self.span_texts = span_texts

    def __len__(self):
        return len(self.span_texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['span_text'] = self.span_texts[idx]
        return item

def split_context_into_spans(context, max_sentences=MAX_SENTENCES, stride=STRIDE):
    """
    Splits the context into overlapping spans based on sentences with variable span sizes (1 to max_sentences).
    
    Args:
        context (str): The long context string.
        max_sentences (int): Maximum number of sentences per span.
        stride (int): Number of sentences to stride between spans.
    
    Returns:
        spans (list): List of span texts, each containing 1 to max_sentences sentences.
    """
    # Split the context into sentences
    sentences = nltk.sent_tokenize(context)
    spans = []
    num_sentences = len(sentences)
    i = 0
    span_size_sequence = [1, 2, 3, 4, 5]
    span_size_index = 0

    while i < num_sentences:
        span_size = span_size_sequence[span_size_index % len(span_size_sequence)]
        span_size = min(span_size, num_sentences - i)  
        span_sentences = sentences[i : i + span_size]
        span_text = ' '.join(span_sentences)
        spans.append(span_text)
        i += (span_size - stride) if (span_size - stride) > 0 else 1 
        span_size_index += 1

    return spans

def prepare_dataloader(question, spans, tokenizer, batch_size=16):
    """
    Prepares the DataLoader for a single question against all spans.

    Args:
        question (str): The question string.
        spans (list): List of context span strings.
        tokenizer: BERT tokenizer.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: DataLoader for (question, span) pairs.
    """
    questions = [question] * len(spans)

    encodings = tokenizer(
        questions,
        spans,
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    dataset = InferenceDataset(encodings, spans)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def rank_spans(model, dataloader, device):
    """
    Runs inference on the DataLoader and ranks the spans based on model predictions.

    Args:
        model: Trained BERT model.
        dataloader (DataLoader): DataLoader for (question, span) pairs.
        device: torch device.

    Returns:
        scores (list): List of probabilities for each span.
        spans (list): List of span texts.
    """
    model.eval()
    all_scores = []
    all_spans = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits  # Shape: (batch_size, num_labels)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()  # Shape: (batch_size,)

            all_scores.extend(probs.tolist())
            all_spans.extend(batch['span_text'])

    return all_scores, all_spans

def get_best_span(scores, spans, top_k=1):
    """
    Retrieves the top_k spans based on the scores.

    Args:
        scores (list): List of probabilities.
        spans (list): List of span texts.
        top_k (int): Number of top spans to retrieve.

    Returns:
        List of tuples: Each tuple contains (span_text, score).
    """
    # Pair each span with its score
    span_scores = list(zip(spans, scores))
    # Sort by score descending
    span_scores.sort(key=lambda x: x[1], reverse=True)
    # Return top_k spans
    return span_scores[:top_k]

def main():
    context = (
        "Artificial Intelligence (AI) has a rich and storied history that spans several decades, "
        "encompassing a wide array of research, development, and application areas. The concept of AI dates "
        "back to ancient myths and stories of artificial beings endowed with intelligence or consciousness by craftsmen "
        "and inventors. However, the formal pursuit of AI as a scientific discipline began in the mid-20th century. "
        "In 1950, British polymath Alan Turing introduced the seminal paper 'Computing Machinery and Intelligence,' "
        "which proposed the famous Turing Test as a measure of machine intelligence. Shortly thereafter, in 1956, the "
        "Dartmouth Conference, organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon, "
        "is widely regarded as the birthplace of AI as a field of study. The attendees of this conference coined the term "
        "'Artificial Intelligence' and set forth the ambitious goal of creating machines that could simulate human intelligence.\n\n"
        "The early years of AI research were characterized by optimism and significant breakthroughs. Pioneers like McCarthy "
        "and Minsky developed the first AI programming languages, such as Lisp, and created foundational AI programs. In 1959, "
        "McCarthy organized the first AI conference, which helped establish AI as a legitimate field of academic inquiry. During "
        "this period, researchers made strides in areas like problem-solving, machine learning, and natural language processing. "
        "The Logic Theorist, developed by Allen Newell and Herbert A. Simon in 1956, is considered one of the first AI programs, "
        "capable of proving mathematical theorems.\n\n"
        "However, the initial enthusiasm for AI was soon met with challenges, leading to periods known as 'AI winters.' "
        "These were times when funding and interest in AI research waned due to unmet expectations and the complexity of the problems. "
        "The first AI winter occurred in the 1970s following the realization that early AI systems were limited in their capabilities. "
        "Subsequent AI winters in the late 1980s and early 1990s were triggered by similar disappointments and the lack of practical applications. "
        "Despite these setbacks, AI research continued, with advancements in specialized areas like expert systems and neural networks.\n\n"
        "The resurgence of AI in the late 1990s and early 2000s was fueled by several factors, including increased computational power, "
        "the availability of large datasets, and the development of more sophisticated algorithms. The advent of big data allowed AI systems to learn from vast amounts of information, "
        "improving their accuracy and efficiency. Machine learning, particularly deep learning, emerged as a powerful approach, enabling the development of models that could recognize patterns, "
        "understand language, and make decisions with minimal human intervention.\n\n"
        "Today's AI landscape is diverse and rapidly evolving, encompassing applications in healthcare, finance, transportation, entertainment, and more. AI-powered systems are used for "
        "diagnosing diseases, predicting market trends, autonomous driving, personalizing content, and even creating art and music. Innovations like reinforcement learning, generative adversarial networks (GANs), "
        "and transformer architectures have pushed the boundaries of what AI can achieve. Additionally, ethical considerations and discussions about the societal impact of AI have become increasingly important, "
        "as the technology continues to integrate into various aspects of daily life.\n\n"
        "Looking forward, the future of AI holds immense potential and challenges. Continued advancements in AI research are expected to lead to more intelligent and autonomous systems, "
        "capable of performing complex tasks with greater efficiency and accuracy. However, as AI systems become more pervasive, issues related to privacy, security, bias, and the ethical use of AI will need to be addressed. "
        "Ensuring that AI technologies are developed and deployed responsibly will be crucial in harnessing their benefits while mitigating potential risks. Collaboration between technologists, policymakers, "
        "and society at large will play a vital role in shaping the trajectory of AI and its role in the future."
    )

    questions = [
        "When did the formal pursuit of Artificial Intelligence as a scientific discipline begin?",
        "What is the Turing Test?",
        "Who organized the Dartmouth Conference and what was its significance?",
        "What was the Logic Theorist and who developed it?",
        "What are AI winters and what caused them?",
        "What factors contributed to the resurgence of AI in the late 1990s and early 2000s?",
        "Name some applications of AI in today's world.",
        "What are some of the ethical considerations related to AI?",
        "What are generative adversarial networks (GANs)?",
        "How can collaboration help in shaping the future of AI?"
    ]

    """tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1  # Binary classification
    )"""

    """tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')
    model = DebertaForSequenceClassification.from_pretrained(
        'microsoft/deberta-base',
        num_labels=1  # Binary classification
    )"""

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=1 
    )

    try:
        model.load_state_dict(torch.load('roberta_f1_73_accuracy_87.pth', map_location=torch.device('cpu')))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Please ensure the model file is in the current directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    model.eval()

    def answer_question(question, span_texts, tokenizer, model, device, batch_size=16, top_k=1):
        dataloader = prepare_dataloader(question, span_texts, tokenizer, batch_size)

        scores, spans = rank_spans(model, dataloader, device)

        top_spans = get_best_span(scores, spans, top_k=top_k)

        return top_spans

    spans = split_context_into_spans(context, max_sentences=MAX_SENTENCES, stride=STRIDE)
    print(f"Total spans created: {len(spans)}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for idx, question in enumerate(questions, 1):
        print(f"\nQuestion {idx}: {question}")
        top_spans = answer_question(
            question=question,
            span_texts=spans,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=16,
            top_k=1
        )

        for rank, (span, score) in enumerate(top_spans, 1):
            print(f"\nTop Answer {rank}:")
            print(f"Score: {score:.4f}")
            print(f"Span: {span}")
            print("-" * 80)

if __name__ == "__main__":
    main()
