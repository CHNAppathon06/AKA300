# Application to map field between a source and target

### Sentence Transformer - Bidirectional Encoder Representations from Transformer
- A Sentence Transformer is a deep learning model that converts sentences, phrases, or words into high-dimensional vector representations (embeddings).
- These embeddings are designed to capture semantic meaning, making them useful for various Natural Language Processing (NLP) tasks, such as text similarity, search, clustering, and classification.

### Approach
- The application leverages two deep learning models `all-MiniLM-L6-v2` and `all-mpnet-base-v2`
- The model is initiated
- The obsolete records are removed in the first method and the nulls are dropped
- Computes cosine similarity
- Computes bert score (excluded - applicable only for a one-to-one mapping)
- Finds the best match. Since the target has more number of fields, we find the best match for every target field from the source and a similarity score is calculated.
- Finally, the results are saved to a csv file.
  
