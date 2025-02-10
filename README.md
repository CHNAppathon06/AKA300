### Application to map fields between a source and target

#### Sentence Transformer - Bidirectional Encoder Representations from Transformer
- A Sentence Transformer is a deep learning model that converts sentences, phrases, or words into high-dimensional vector representations (embeddings).
- These embeddings are designed to capture semantic meaning, making them useful for various Natural Language Processing (NLP) tasks, such as text similarity, search, clustering, and classification.
  
#### Approach
- The user can pass any suitable model as an argument -> `mapper.py`
- The model is initiated
- The obsolete records are removed and the nulls are dropped
- Cosine similarity is calculated.
- Bert score (excluded - applicable only for a one-to-one mapping)
- For each field from the source the best match is identified. Since the target has more number of fields, we find the best match for every target field from the source and a similarity score is calculated. (source <-> target)
- Finally, the results are saved to a csv file.
