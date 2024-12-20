"""Calculate and Visualise metrics"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score


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
    
    acc_input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
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


def extract_SE(model_results, type):
    """Extracts the semantic entropy (SE) values from the results of multiple models and prepares the data for visualization

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
            models_names.append(f"{type} {model.capitalize()} {size}")
            only_datasets = [list(item.values())[0] for item in model_results[model][size]]
            results += [dataset["semantic_entropy"] for dataset in only_datasets]
    
    return models_names, results


def visualise_auroc_results(models_names, datasets_names, results, colours):
    """Creates a bar plot to visualize AUROC scores across different models and datasets

    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which AUROC scores are computed
        results (list): A nested list containing AUROC scores for each model and dataset
        colours (dict): A dictionary mapping model names to their respective colors

    Output:
        Saves the plot as `results/figures/auroc_scores_plot.png`

    Returns:
        None
    """

    spacing = 1.6  # Adjust this value to control the space between datasets
    x = np.arange(0, len(datasets_names) * spacing, spacing)
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, result) in enumerate(zip(models_names, results)):
        colour = colours.get(model, 'grey')
        ax.bar(x + i * width, result, width, label=model, color=colour)

    ax.set_xlabel("Datasets")
    ax.set_ylabel("AUROC Score")
    ax.set_title("AUROC Scores for Different Models and Datasets")
    ax.set_xticks(x + width * (len(models_names) - 1) / 2)
    ax.set_xticklabels(datasets_names)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", title="Models", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("results/figures/auroc_scores_plot.png")
    plt.show()


def visualise_SE_distribution(models_names, datasets_names, results, colours):
    """Creates a grouped box plot to visualize the distribution of semantic entropy (SE) values for various models across datasets

    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which AUROC scores are computed
        results (list): A nested list containing AUROC scores for each model and dataset
        colours (dict): A dictionary mapping model names to their respective colors

    Output:
        Saves the plot as `results/figures/SE_distribution.png`

    Returns:
        None
    """

    groups = len(datasets_names)
    boxes_per_group = int(len(results)/groups)

    positions = [] # Set positions for the boxes (manually group them)
    for i in range(groups):
        positions += [i * (boxes_per_group + 1) + j for j in range(1, boxes_per_group + 1)]

    plt.figure(figsize=(16, 6))
    unique_models = list(set(models_names))
    unique_models.sort()

    # Create the boxplot - plot one box at a time with the correct color
    for i in range(groups):  # Loop through each dataset group
        for j in range(boxes_per_group):  # Loop through boxes within the group
            idx = i * boxes_per_group + j  # Calculate overall index
            colour = colours[models_names[j]]  # Get the color for this model
            plt.boxplot(
                [results[idx]],  # Boxplot for this model's result
                positions=[positions[idx]],  # Position for this box
                patch_artist=True,
                boxprops=dict(facecolor=colour, color=colour),
                medianprops=dict(color="black"),
                widths=0.5
            )

    plt.xticks([i * (boxes_per_group + 1) + (boxes_per_group / 2) for i in range(groups)], datasets_names)

    legend_patches = [mpatches.Patch(color=colours[model], label=model) for model in unique_models]
    plt.legend(handles=legend_patches, loc="upper left", title="Models", bbox_to_anchor=(1, 1)) # Place legend outside to the right
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(right=0.7)  # Shrink the plot to make space for the legend
    plt.title('Distribution of Semantic Entropy')
    plt.xlabel('Datatsets')
    plt.ylabel('Semantic Entropy')
    plt.savefig('results/figures/SE_distribution.png', bbox_inches='tight')
    plt.show()


def visualise_SE_mean_std(models_names, datasets_names, results, colours):
    """Visualizes the mean and standard deviation of semantic entropy (SE) values for different 
        models and datasets using line plots with std bands

    Parameters:
        models_names (list): A list of model names corresponding to the results
        datasets_names (list): A list of dataset names for which AUROC scores are computed
        results (list): A nested list containing AUROC scores for each model and dataset
        colours (dict): A dictionary mapping model names to their respective colors

    Output:
        Saves the plot as `results/figures/SE_mean_std.png`

    Returns:
        None
    """

    grouped_results = {dataset: {} for dataset in datasets_names}

    index = 0
    for dataset in datasets_names:
        for model in models_names:
            grouped_results[dataset][model] = results[index]
            index += 1

    means = []
    stds = []
    for dataset in datasets_names:
        dataset_means = []
        dataset_stds = []
        for model in models_names:
            model_results = grouped_results[dataset][model]
            dataset_means.append(np.mean(model_results))
            dataset_stds.append(np.std(model_results))
        means.append(dataset_means)
        stds.append(dataset_stds)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets_names))  # Dataset positions

    for i, model in enumerate(models_names):
        mean_values = [means[j][i] for j in range(len(datasets_names))]
        std_values = [stds[j][i] for j in range(len(datasets_names))]
        ax.plot(x, mean_values, label=model, marker='o', linestyle='-', color=colours[model])
        ax.fill_between(x, 
                        [m - s for m, s in zip(mean_values, std_values)], 
                        [m + s for m, s in zip(mean_values, std_values)], 
                        color=colours[model], alpha=0.4)

    ax.set_xlabel("Datasets")
    ax.set_ylabel("Semantic Entropy (Mean ± Std)")
    ax.set_title("Mean and Standard Deviation of SE for Models")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_names)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", title="Models", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("results/figures/SE_mean_std.png")
    plt.show()