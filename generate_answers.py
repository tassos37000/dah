"""TODO desc"""

from torch.cuda import empty_cache
from tqdm import tqdm
from my_utils.semantic_entropy import gen_responses_probs, is_entailment_transformer, is_entailment_llm, cluster_responses, calculate_sem_entr
from my_utils.metrics import assess_acc

def generate_answers(datasets, data_transformer_path, data_llm_path, llm_model, llm_tokenizer, 
                     entail_transformer_model, entail_transformer_tokenizer, entail_llm_model, entail_llm_tokenizer):
    """
    TODO desc
    """
    for dataset in datasets:
        all_responses = []
        all_acc_resp = []
        all_labels = []
        all_transformer_clusters = []
        all_transformer_sem_entr = []
        all_llm_clusters = []
        all_llm_sem_entr = []

        intro_promt = "Answer the following question in a single brief but complete sentence. "
        print(f"\nGenerating responses for {dataset.info.description} dataset...")

        for i in tqdm(range(len(dataset))):
            # Generate responses for Semantic Entropy and Accuracy
            responses = gen_responses_probs(llm_model, llm_tokenizer, intro_promt + dataset[i]["question"])
            empty_cache()
            acc_response = gen_responses_probs(llm_model, llm_tokenizer, intro_promt + dataset[i]["question"], number_responses=1, temperature=0.1)
            empty_cache()
            acc_response_text = llm_tokenizer.decode(acc_response["sequences"][0], skip_special_tokens=True)
            empty_cache()
            label = assess_acc(llm_model, llm_tokenizer, dataset[i]["question"], str(dataset[i]["answers"]["text"]), acc_response_text)
            empty_cache()
            all_responses.append(responses)
            all_acc_resp.append(acc_response)
            all_labels.append(label)

            # TODO move them in different function so you can free-up memory by deleting the main LLM
            # Calculate transformer semantic entropy
            transformer_clusters = cluster_responses(responses, llm_tokenizer, is_entailment_transformer, entail_transformer_model, entail_transformer_tokenizer, dataset[i]["question"])
            empty_cache()
            transformer_sem_entr = calculate_sem_entr(transformer_clusters, responses["sequences_probabilities"])
            empty_cache()
            all_transformer_clusters.append(transformer_clusters)
            all_transformer_sem_entr.append(transformer_sem_entr)

             # Calculate llm semantic entropy
            llm_clusters = cluster_responses(responses, llm_tokenizer, is_entailment_llm, entail_llm_model, entail_llm_tokenizer, dataset[i]["question"])
            empty_cache()
            llm_sem_entr = calculate_sem_entr(llm_clusters, responses["sequences_probabilities"])
            empty_cache()
            all_llm_clusters.append(llm_clusters)
            all_llm_sem_entr.append(llm_sem_entr)
            
        # Save results to dataset
        dataset = dataset.add_column("generated_answers", all_responses)
        dataset = dataset.add_column("generated_answer_acc", all_acc_resp)
        dataset = dataset.add_column("labels", all_labels)
        dataset = dataset.add_column("clusters", all_transformer_clusters)
        dataset = dataset.add_column("semantic_entropy", all_transformer_sem_entr)
        dataset.save_to_disk(data_transformer_path + dataset.info.description)

        dataset = dataset.remove_columns(["clusters", "semantic_entropy"])
        dataset = dataset.add_column("clusters", all_llm_clusters)
        dataset = dataset.add_column("semantic_entropy", all_llm_sem_entr)
        dataset.save_to_disk(data_llm_path + dataset.info.description)

        del all_responses, all_acc_resp, all_transformer_clusters, all_transformer_sem_entr, all_llm_clusters, all_llm_sem_entr, all_labels
        empty_cache()   