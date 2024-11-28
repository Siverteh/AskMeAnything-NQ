import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, RobertaForSequenceClassification, BertForSequenceClassification, BertTokenizer, RobertaTokenizer
import torch.nn.functional as F

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
    #tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    """model = DebertaForSequenceClassification.from_pretrained(
            'microsoft/deberta-base',
            num_labels=2
        )"""

    model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=2,
            hidden_dropout_prob=dropout_prob
        )

    """model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            hidden_dropout_prob=dropout_prob
        )"""
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  

    return model, tokenizer

def predict_yes_no(model, tokenizer, question, context, max_length=256):
    """
    Predicts a yes/no answer to a question based on the provided context.

    Args:
        model: The DeBERTa model.
        tokenizer: The DeBERTa tokenizer.
        question (str): The question to answer.
        context (str): The context containing the answer.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        str: 'YES' or 'NO' based on the model's prediction.
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
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    answer = 'YES' if predicted_class == 1 else 'NO'

    return answer

if __name__ == "__main__":
    #BEST DeBERTa MODEL, DROPOUT FOR MODEL: 0.34438747129324265
    #model_path = 'DeBERTa_82.pth'  # Replace with your model's path if different
    #model, tokenizer = load_model(model_path, 0.168070018031781)
    #BEST RoBERTa MODEL, DROPOUT FOR MODEL: 0.16888306619130625
    model_path = 'RoBERTa_74.pth'
    model, tokenizer = load_model(model_path, 0.168070018031781)

    #BEST BaseBERT MODEL, DROPOUT FOR MODEL: 0.307584268895457
    #model_path = 'BaseBERT_76.pth'
    #model, tokenizer = load_model(model_path, 0.307584268895457)

    samples = [
    {
        'context': """
        The BBC announced on 2 March 2017 that there would be no further series.
        """,
        'question': "Will there be another series of The Coroner?",
        'actual_answer': "NO"
    },
    {
        'context': """
        In 2008, a Home Office circular made clear suspects must receive a written explanation of the implications before accepting a caution, 
        to meet the informed consent obligation, and provided a new form to be signed by the offender which explained in considerable detail the consequences.
        """,
        'question': "Do I have to sign a police caution?",
        'actual_answer': "YES"
    },
    {
        'context': """
        The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood,
        and other materials, generally built along an east-to-west line across the historical northern borders
        of China to protect the Chinese states and empires against the raids and invasions of the various
        nomadic groups of the Eurasian Steppe.
        """,
        'question': "Is the Great Wall of China visible from the moon?",
        'actual_answer': "NO"
    },
    {
        'context': """
        Although the continental landmasses listed below are not normally called islands (by definition a landmass cannot be both an island and a continent), 
        they are, in fact, land entirely surrounded by water. In effect, they are enormous islands and are shown here for that reason. 
        The figures are approximations and are for the continental mainland only.
        """,
        'question': "Is Australia the largest island in the world?",
        'actual_answer': "NO"
    },
    {
        'context': """
        Alaska has only been hit by 2 tornadoes between 1950 and 2006 (the first being November 4, 1959, and the other was August 2, 2005); both were only F0. 
        Hawaii has only been hit by 49 tornadoes since 1950.
        """,
        'question': "Has there ever been a tornado in Alaska?",
        'actual_answer': "YES"
    },
    {
        'context': """
        Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. 
        It is a gas giant with an average radius about nine times that of Earth and has the most extensive rings of any planet.
        It is not the largest planet in our Solar System.
        """,
        'question': "Is Saturn the largest planet in the Solar System?",
        'actual_answer': "NO"
    },
    {
        'context': """
        The American Civil War, fought between 1861 and 1865, was a conflict between the United States (the Union) and eleven Southern states 
        that seceded from the Union to form the Confederate States of America.
        """,
        'question': "Did the American Civil War start in 1861?",
        'actual_answer': "YES"
    },
    {
        'context': """
        Jupiter's moon Europa is one of the largest moons in the Solar System and is primarily composed of silicate rock and has a water-ice crust. 
        It is slightly smaller than Earth's moon and orbits Jupiter every 3.5 days.
        """,
        'question': "Is Europa the largest moon of Jupiter?",
        'actual_answer': "NO"
    },
    {
        'context': """
        The Amazon River in South America is the largest river by discharge volume of water in the world, and it is second in length only to the Nile River.
        The Amazon basin is the largest drainage basin in the world.
        """,
        'question': "Is the Amazon River the longest river in the world?",
        'actual_answer': "NO"
    },
    {
        'context': """
        The Mariana Trench is the deepest part of the world's oceans, located in the western Pacific Ocean. 
        It reaches a maximum-known depth of about 10,994 meters (36,070 feet) at the Challenger Deep.
        """,
        'question': "Is the Mariana Trench the deepest part of the ocean?",
        'actual_answer': "YES"
    }
]


for idx, sample in enumerate(samples):
    context = sample['context']
    question = sample['question']
    actual_answer = sample['actual_answer']
    
    predicted_answer = predict_yes_no(model, tokenizer, question, context)
    
    print(f"Sample {idx + 1}:")
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Actual Answer: {actual_answer}")
    print("-" * 50)

