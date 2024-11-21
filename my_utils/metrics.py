"""Calculate metrics"""

import numpy as np
# from torch.cuda import is_available
from config import DEVICE
from sklearn.metrics import roc_auc_score

# device = "cuda" if is_available() else "cpu"

def assess_acc(model, tokenizer, question, answers, response):
    """
    TODO desc
    """

    prompt = (f"We are assessing the quality of answers to the following question: {question}\n"
              f"The expected answer is: {answers}\n"
              f"The proposed answer is: {response}\n"
              f"Within the context of the question, does the proposed answer mean the same as the expected answer?\n"
              f"Respond only with yes or no.")
    
    acc_input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = acc_input_ids["input_ids"].shape[1]
    output = model.generate(**acc_input_ids, max_new_tokens=256, return_dict_in_generate=True)
    text_res = tokenizer.decode(output.sequences[0, input_length:], skip_special_tokens=True).lower()
    
    return 1 if "yes" in text_res else 0


def calculate_auroc(datasets):
    """
    TODO desc
    """

    auroc_list = []
    for d in datasets:
        auroc = roc_auc_score(1-np.array(d["labels"]), d["semantic_entropy"])
        auroc_list.append(auroc)
        print(f"{d.info.description:20} dataset: {auroc:8.6f}")

    return auroc_list