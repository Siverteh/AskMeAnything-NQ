import torch
from transformers import DebertaTokenizerFast, DebertaForQuestionAnswering, RobertaForQuestionAnswering, RobertaTokenizerFast

def load_model(model_path, dropout_prob=0.34438747129324265):
    """
    Loads the trained DeBERTa model and tokenizer.

    Args:
        model_path (str): Path to the saved model weights.
        dropout_prob (float): The dropout probability used during training.

    Returns:
        model: The loaded DeBERTa model.
        tokenizer: The DeBERTa tokenizer.
    """
    #tokenizer = DebertaTokenizerFast.from_pretrained('microsoft/deberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


    """model = DebertaForQuestionAnswering.from_pretrained(
        'microsoft/deberta-base',
        num_labels=2
    )"""
    model = RobertaForQuestionAnswering.from_pretrained(
        'roberta-base',
        num_labels=2
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  

    return model, tokenizer

def predict_answer(model, tokenizer, question, context, max_length=256):
    """
    Predicts the answer span in the context for a given question.

    Args:
        model: The DeBERTa model.
        tokenizer: The DeBERTa tokenizer.
        question (str): The question to answer.
        context (str): The context containing the answer.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        str: The predicted answer text from the context.
    """
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        padding='max_length',
        truncation='only_second',
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    answer_tokens = inputs['input_ids'][0][start_idx: end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

if __name__ == "__main__":
    model_path = 'RoBERTa_64_EM_78_F1.pth'  
    model, tokenizer = load_model(model_path, 0.12603647331464332)

    samples = [
    {
        'context': """
        The Galapagos Islands, located in the Pacific Ocean off the coast of Ecuador, are famous for their unique and diverse ecosystems. 
        The islands were studied extensively by Charles Darwin during his voyage on the HMS Beagle in 1835. Observing the differences 
        between species on different islands, particularly the finches, led Darwin to develop his theory of evolution by natural selection. 
        This theory, later published in his seminal work 'On the Origin of Species' in 1859, revolutionized biology and provided 
        a scientific explanation for the diversity of life on Earth.
        """,
        'question': "On which ship did Charles Darwin travel to the Galapagos Islands?",
        'actual_answer': "HMS Beagle"
    },
    {
        'context': """
        The Silk Road was an ancient network of trade routes that connected the East and West, allowing for the exchange of goods, culture, 
        and ideas. Originating in China, it extended across Asia to the Mediterranean Sea, enabling the spread of silk, spices, and other 
        valuable commodities. The Silk Road also facilitated the exchange of knowledge, religion, and technology, influencing many cultures 
        along its path. The route declined in importance with the rise of maritime trade in the 15th century, but its impact on world history 
        remains significant to this day.
        """,
        'question': "Which valuable commodity was primarily traded along the Silk Road from China?",
        'actual_answer': "Silk"
    },
    {
        'context': """
        The Great Fire of London in 1666 was a major conflagration that swept through the central parts of the English capital. 
        The fire started in a bakery on Pudding Lane and quickly spread, consuming thousands of homes, churches, and other buildings. 
        While the fire destroyed much of the medieval city, it also led to improvements in building regulations and the reconstruction 
        of London with fire-resistant materials. Sir Christopher Wren, an acclaimed architect, played a central role in the city's 
        rebuilding, designing many of London's most famous structures, including St. Paul's Cathedral.
        """,
        'question': "Where did the Great Fire of London begin?",
        'actual_answer': "Pudding Lane"
    },
    {
        'context': """
        The Suez Canal, an artificial waterway in Egypt, connects the Mediterranean Sea to the Red Sea and provides the shortest 
        maritime route between Europe and the lands lying around the Indian and western Pacific oceans. Completed in 1869, 
        the canal has been a crucial route for international trade, allowing ships to avoid the long and hazardous journey around 
        the southern tip of Africa. Its control has often been a point of international contention, most notably during the 
        Suez Crisis in 1956, when Egypt nationalized the canal, leading to a military intervention by Britain, France, and Israel.
        """,
        'question': "Which two seas does the Suez Canal connect?",
        'actual_answer': "Mediterranean Sea and Red Sea"
    },
    {
        'context': """
        The Battle of Waterloo, fought on June 18, 1815, marked the end of the Napoleonic Wars. This decisive battle pitted 
        the French army, led by Napoleon Bonaparte, against the Seventh Coalition, which included British and Prussian forces 
        commanded by the Duke of Wellington and Gebhard Leberecht von Blücher. Napoleon's defeat at Waterloo ended his rule 
        as Emperor of the French and led to his exile to the island of Saint Helena. The battle also signified the beginning 
        of a new era in European politics and the end of Napoleon's influence on the continent.
        """,
        'question': "Who commanded the British forces at the Battle of Waterloo?",
        'actual_answer': "Duke of Wellington"
    },
    {
        'context': """
        The Panama Canal, a major engineering marvel, is an artificial waterway that connects the Atlantic and Pacific Oceans 
        through the narrow Isthmus of Panama. Opened in 1914, the canal drastically reduced the travel time for ships, allowing 
        them to bypass the lengthy and dangerous route around the southern tip of South America. The canal consists of a series 
        of locks that raise and lower ships to accommodate the different sea levels. Originally constructed by the United States, 
        control of the canal was handed over to Panama in 1999, following the terms of the Torrijos-Carter Treaties.
        """,
        'question': "Which two oceans does the Panama Canal connect?",
        'actual_answer': "Atlantic Ocean and Pacific Ocean"
    }
]

    for idx, sample in enumerate(samples):
        context = sample['context']
        question = sample['question']
        actual_answer = sample['actual_answer']
        
        predicted_answer = predict_answer(model, tokenizer, question, context)
        
        print(f"Sample {idx + 1}:")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Actual Answer: {actual_answer}")
        print("-" * 50)
