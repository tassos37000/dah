"""Semantic Entropy Utilities."""

import torch
import torch.nn.functional as F
import numpy as np
from config import DEVICE

# device = "cuda" if torch.cuda.is_available() else "cpu"


def gen_responses_probs(model, tokenizer, question, number_responses=10, temperature=1.0):
    """ Generates 10 responses with high temeperature

    Parameters:
        model (AutoModelForCausalLM): The language generation model
        tokenizer (AutoTokenizer): The tokenizer for the model
        question (str): The input question to generate responses for
        number_responses (int): The number of responses to generate, default 10
        number_responses (float): The number used to modulate the next token probabilities, default 1.0

    Returns:
        dict: The deafult dictionary returned from generate() with token and sequence probabilities
    """

    input_ids = tokenizer(question, return_tensors="pt").to(DEVICE)
    input_length = input_ids['input_ids'].shape[1]

    outputs_high_temp = model.generate(
        **input_ids,
        max_new_tokens=64,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        top_k=50,                  # top-K sampling
        top_p=0.9,                 # nucleus sampling
        temperature=temperature,
        num_return_sequences=number_responses
    )

    sequence_token_probabilities = []
    generated_answer_tokens = []
    for idx in range(number_responses):
        generated_tokens = outputs_high_temp.sequences[idx, input_length:] # Only keep the generated tokens by slicing out the input question tokens
        generated_answer_tokens.append(generated_tokens.cpu().tolist())
        
        probabilities = [F.softmax(score, dim=-1) for score in outputs_high_temp.scores] # Calculate probabilities for each token in the generated response
        token_probabilities = []
        for i, token_id in enumerate(generated_tokens):
            token_prob = probabilities[i][idx, token_id].item()  # [idx, token_id] for batch dimension
            if token_prob > 0:
                token_probabilities.append(token_prob)
        sequence_token_probabilities.append(token_probabilities)

    outputs = {
        "sequences": generated_answer_tokens,
        # "tokens_probabilities": sequence_token_probabilities,
        "sequences_probabilities": [-np.sum(np.log(prob)) / len(prob) for prob in sequence_token_probabilities], 
    }

    return outputs


def is_entailment_transformer(model, tokenizer, premise, hypothesis, question=""):
    """ Checks if two sentences have bidirectional entailment
    
    Parameters:
        model (AutoModelForSequenceClassification): The Sequence classification model
        tokenizer (AutoTokenizer): The tokenizer for the Sequence classification model
        premise (string): The 
        hypothesis (string): TODO desc
        question (string):

    Returns:
        boolean: True if there is bidirectional entailment else False
    """

    # premise -> hypothesis
    input_ids = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = model(**input_ids)
    logits = outputs.logits # Get the entailment scores
    entailment_prob = torch.softmax(logits, dim=1)
    label_pre_hypo = torch.argmax(entailment_prob, dim=1).item()

    # hypothesis -> premise
    input_ids = tokenizer(hypothesis, premise, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = model(**input_ids)
    logits = outputs.logits # Get the entailment scores
    entailment_prob = torch.softmax(logits, dim=1)
    label_hypo_pre = torch.argmax(entailment_prob, dim=1).item()

    return (label_pre_hypo == 2) and (label_hypo_pre == 2)


def is_entailment_llm(model, tokenizer, premise, hypothesis, question):
    """
    TODO desc
    """

    prompt = (f"We are evaluating answers to the question {question}\n"
              f"Here are two possible answers:\n"
              f"Possible Answer 1: {premise}\n"
              f"Possible Answer 2: {hypothesis}\n"
              f"Does Possible Answer 1 semantically entail Possible Answer 2?\n"
              f"Respond with entailment, contradiction, or neutral.")
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
        ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    entail_input_ids = tokenizer([text], return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(**entail_input_ids, max_new_tokens=256)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(entail_input_ids.input_ids, generated_ids)    ]
    text_res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return True if "entailment" in text_res else False


def cluster_responses(responses, llm_tokenizer, is_entailment, entail_model, entail_tokenizer, question=""):
    """ Create the clusters from the responses
    
    Parameters:
        responses (list): The Sequence classification model
        llm_tokenizer (AutoTokenizer): The tokenizer for the LLM model
        is_entailment (function): The function that will be used to asses the enatilment
        entail_model (AutoModelForCausalLM): The Sequence classification model
        entail_tokenizer (AutoTokenizer): The tokenizer for the Sequence classification model
        question (string): TODO desc

    Returns:
        List: A list where each cluster is represented by another list with the index of the responses
    """

    clusters = [[0]]
    for i in range(1, len(responses['sequences'])):
        for c in clusters:
            response_text = llm_tokenizer.decode(responses['sequences'][i], skip_special_tokens=True)
            cluster_text = llm_tokenizer.decode(responses['sequences'][c[0]], skip_special_tokens=True)
            if is_entailment(entail_model, entail_tokenizer, response_text, cluster_text, question):
                c.append(i)
                break
            else:
                clusters.append([i])
                break  
    
    return clusters


def calculate_sem_entr(clusters, sequences_prob):
    """ Calculates Semantic Entropy from clustered responses

    Parameters:
        clusters (List): A list where each cluster is represented by another list with the index of the responses
        sequences_prob (List): A list with the sequence probability of each response

    Returns:
        float: The Semantic Entropy of all clusters
    """

    sem_entr = 0.0
    norm_prob = sum(sequences_prob)
    
    for cluster in clusters:
        cluster_prob = sum(sequences_prob[i] for i in cluster) / norm_prob
        sem_entr += cluster_prob * np.log10(cluster_prob)
    
    return -sem_entr