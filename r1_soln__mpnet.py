# Compares two fields from soure and target and computes the similarity score

import pandas as pd
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from bert_score import score

# Load Sentence Transformer Model for Cosine Similarity
model = SentenceTransformer("all-mpnet-base-v2")

# Define file paths
source_path = Path(r"D:\appathon_parser\Files for App-a-thon\requirement 1 mapping\vendor_input_format.csv")
target_path = Path(r"D:\appathon_parser\Files for App-a-thon\requirement 1 mapping\customer_standard_format.csv")

def read_and_clean_data(input_path):
    """
    Reads a CSV file, removes unused fields, and returns a list of business names.
    """
    data = pd.read_csv(input_path, encoding="ISO-8859-1")
    data_filtered = data[~data.astype(str).apply(lambda row: row.str.contains("Field is not currently used", 
                                                                              case=False, na=False)).any(axis=1)]
    return data_filtered['Business Name'].dropna().tolist()  # Drop NaN values

def compute_cosine_similarity(vendor_list, customer_list):
    """
    Computes cosine similarity between customer and vendor field names.
    """
    vendor_embeddings = model.encode(vendor_list, convert_to_tensor=True)
    customer_embeddings = model.encode(customer_list, convert_to_tensor=True)
    return util.pytorch_cos_sim(customer_embeddings, vendor_embeddings)  # Similarity matrix

def compute_bertscore(vendor_list, customer_list):
    """
    Computes BERTScore between customer and vendor field names.
    """
    P, R, F1 = score(customer_list, vendor_list, lang="en", verbose=True)
    return F1.tolist()  # Return F1 scores as a list

def find_best_matches(customer_list, vendor_list, similarity_matrix):
    """
    Finds the best match for each customer field from the vendor list using both Cosine Similarity and BERTScore.
    """
    best_matches = []
    for i, customer_field in enumerate(customer_list):
        best_match_idx = similarity_matrix[i].argmax().item()
        best_match_target = vendor_list[best_match_idx]
        best_match_score = similarity_matrix[i][best_match_idx].item()
        # bert_score_value = bert_scores[i * len(vendor_list) + best_match_idx]

        best_matches.append((customer_field, best_match_target, round(best_match_score, 4)))
    return best_matches

def save_results_to_csv(results, output_filename):
    """
    Writes the best match results to a CSV file with both Cosine Similarity.
    """
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Customer Field", "Vendor Field", "Cosine Similarity"])
        writer.writerows(results)
    print(f"Results saved to {output_filename}")

# Main Execution Flow
if __name__ == "__main__":
    vendor_fields = read_and_clean_data(source_path)
    customer_fields = read_and_clean_data(target_path)

    similarity_matrix = compute_cosine_similarity(vendor_fields, customer_fields)
    # bert_scores = compute_bertscore(vendor_fields, customer_fields)

    best_matches = find_best_matches(customer_fields, vendor_fields, similarity_matrix)

    save_results_to_csv(best_matches, 'r1_soln__mpnet.csv')
