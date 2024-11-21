"""Data Utilities."""
import hashlib
import datasets
import random


def sample_ds(dataset, n_samples, seed, name):
    """ Selects random samples from a dataset
    
    Parameters:
        dataset (Dataset): The dataset
        n_samples (int): The number of samples
        seed (int): The seed for random library
        name (string): The name of the dataset

    Returns:
        Dataset: Returns a Dataset with size equal to the n_samples and name is saved in the .info.description
    """

    random.seed(seed)
    random_indices = [random.randint(0, dataset.num_rows) for _ in range(n_samples)]
    dataset = dataset.select(random_indices)
    dataset.info.description = name
    print("Dataset: ", name)
    print(dataset, "\n")
    return dataset


def load_ds(dataset_name, seed):
    """ Loads datasets from Hugging Face
    
    Parameters:
        dataset_name (string): Name of the dataset in the Hugging Face library
        seed (int): The seed for random library

    Returns:
        (Dataset, Dataset): Returns the train and validation datasets

    Parts of function from https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/uncertainty/data/data_utils.py
    """

    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = train_dataset.map(reformat)
        validation_dataset = validation_dataset.map(reformat)

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = train_dataset.map(reformat)
        validation_dataset = validation_dataset.map(reformat)

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    else:
        raise ValueError

    return train_dataset, validation_dataset