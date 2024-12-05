"""Calculate metrics"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from config import DEVICE


def assess_acc(model, tokenizer, question, answers, response):
    """Assesses the semantic equivalence between a proposed response and the expected answer for a given question

    Parameters:
        model (AutoModelForCausalLM): A language model used to evaluate the response
        tokenizer (AutoTokenizer): The tokenizer associated with the language model
        question (str): The question to which the answers are related
        answers (str): The ground-truth answer(s) to the question
        response (str): The proposed answer to be assessed

    Returns:
        int: Returns 1 if the model determines the response is equivalent to the expected answer, otherwise 0
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
    """Computes the AUROC for each dataset

    Parameters:
        datasets (list): A list of datasets, where each dataset contains labels (binary ground-truth) 
                         and semantic entropy values for the responses

    Returns:
        list: A list of AUROC scores, one for each dataset
    """

    auroc_list = []
    for d in datasets:
        auroc = roc_auc_score(1-np.array(d["labels"]), d["semantic_entropy"])
        auroc_list.append(auroc)
        print(f"{d.info.description:20} dataset: {auroc:8.4f}")

    return auroc_list


def auroc_entail_models(model_results, type):
    """Computes AUROC scores for different entailment models and their respective sizes across datasets

    Parameters:
        model_results (dict): A nested dictionary containing results for various models and their sizes
                              Structure example:
                              {
                                  "model1": {
                                      "0.5B": [{dataset_name: dataset_object}, ...],
                                      "3.0B": [{dataset_name: dataset_object}, ...],
                                  },
                                  "model2": {
                                      "14.0B": [{dataset_name: dataset_object}, ...],
                                      "30.0B": [{dataset_name: dataset_object}, ...],
                                  },
                                  ...
                              }
        type (str): The type of entailment being evaluated (e.g., "LLM" for language models or "Transformer")

    Returns:
        tuple:
            - models_names (list): A list of strings, where each string represents a model and its size (e.g., "LLM Model1 Small")
            - results (list): A list of AUROC scores for each model and size combination, computed across datasets
    """

    models_names = []
    results = []

    for model in model_results.keys():
        for size in model_results[model].keys():
            print(f"\nAUROC scores for {type} {model.capitalize()} {size}")
            models_names.append(f"{type} {model.capitalize()} {size}")
            only_datasets = [list(item.values())[0] for item in model_results[model][size]]
            results.append(calculate_auroc(only_datasets))
    
    return models_names, results


def visualise_results(models_names, datasets_names, results):
    """Creates a bar plot to visualize AUROC scores across different models and datasets

    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which AUROC scores are computed
        results (list): A nested list containing AUROC scores for each model and dataset

    Output:
        Saves the plot as `results/figures/auroc_scores_plot.png`

    Returns:
        None
    """

    spacing = 1.2  # Adjust this value to control the space between datasets
    x = np.arange(0, len(datasets_names) * spacing, spacing)
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, result) in enumerate(zip(models_names, results)):
        ax.bar(x + i * width, result, width, label=model)

    ax.set_xlabel("Datasets")
    ax.set_ylabel("AUROC Score")
    ax.set_title("AUROC Scores for Different Models and Datasets")
    ax.set_xticks(x + width * (len(models_names) - 1) / 2)
    ax.set_xticklabels(datasets_names)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("results/figures/auroc_scores_plot.png")
    plt.show()