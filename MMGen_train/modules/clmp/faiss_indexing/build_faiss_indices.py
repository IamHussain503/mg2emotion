# # import numpy as np
# # import faiss
# # import time
# # import os

# # def build_hnsw_index(data, M=32, efConstruction=128):
# #     """
# #     Build HNSW index.

# #     Args:
# #         data (np.ndarray): Dataset with shape (n_samples, d).
# #         M (int): Number of connections in HNSW.
# #         efConstruction (int): ef parameter during construction.

# #     Returns:
# #         faiss.IndexHNSWFlat: Built HNSW index.
# #     """
# #     d = data.shape[1]
# #     index = faiss.IndexHNSWFlat(d, M)
# #     index.hnsw.efConstruction = efConstruction

# #     start_time = time.time()
# #     index.add(data)
# #     end_time = time.time()

# #     print(f"HNSW index construction time: {end_time - start_time:.2f} seconds")
# #     return index

# # def evaluate_index(audio_index, melody_index, audio_queries, k=1):
# #     """
# #     Evaluate index performance.

# #     Args:
# #         audio_index (faiss.Index): Audio index.
# #         melody_index (faiss.Index): Melody index.
# #         audio_queries (np.ndarray): Query vectors with shape (n_queries, d).
# #         k (int): Number of nearest neighbors.

# #     Returns:
# #         dict: Evaluation results with search times.
# #     """
# #     # Evaluate audio to audio query
# #     audio_to_audio_start_time = time.time()
# #     _, _ = audio_index.search(audio_queries, k)
# #     audio_to_audio_search_time = time.time() - audio_to_audio_start_time

# #     # Evaluate audio to melody query
# #     audio_to_melody_start_time = time.time()
# #     _, _ = melody_index.search(audio_queries, k)
# #     audio_to_melody_search_time = time.time() - audio_to_melody_start_time

# #     return {
# #         'k': k,
# #         'audio_to_audio_search_time': audio_to_audio_search_time,
# #         'audio_to_melody_search_time': audio_to_melody_search_time,
# #     }

# # def save_index(index, index_path):
# #     """
# #     Save FAISS index to specified path.

# #     Args:
# #         index (faiss.Index): Index to save.
# #         index_path (str): File path to save the index.
# #     """
# #     faiss.write_index(index, index_path)
# #     print(f"Index saved to: {index_path}")

# # if __name__ == "__main__":
# #     # Load data
# #     melody_data = np.load('/root/m2music/data/embeddings/melody_embeddings.npy')
# #     print("melody_data loaded")
# #     audio_data = np.load('/root/m2music/data/embeddings/audio_embeddings.npy')
# #     print("audio_data loaded")
# #     audio_queries = np.load('/root/m2music/data/embeddings/text_embeddings.npy')
# #     print("audio_queries loaded")
    
# #     # HNSW parameters
# #     M = 32
# #     efConstruction = 80
# #     k = 1

# #     print("Building melody HNSW index...")
# #     melody_index = build_hnsw_index(melody_data, M=M, efConstruction=efConstruction)
    
# #     print("Building audio HNSW index...")
# #     audio_index = build_hnsw_index(audio_data, M=M, efConstruction=efConstruction)

# #     index_info = f"HNSW, M: {M}, efConstruction: {efConstruction}"
# #     file_suffix = "hnsw"

# #     # Run validation
# #     print(f"Running validation with k={k}...")
# #     validation_result = evaluate_index(audio_index, melody_index, audio_queries, k=k)

# #     # Output validation results
# #     print(f"\nIndex type: {index_info}, k: {k}")
# #     print(f"Audio to audio search time: {validation_result['audio_to_audio_search_time']:.6f} seconds")
# #     print(f"Audio to melody search time: {validation_result['audio_to_melody_search_time']:.6f} seconds")

# #     # Save index
# #     save_path = '/root/m2music/data/faiss'

# #     melody_index_path = os.path.join(save_path, f'audio_2_melody_{file_suffix}.faiss')
# #     print("Saving melody HNSW index...")
# #     save_index(melody_index, melody_index_path)


# import numpy as np
# import faiss
# import time
# import os



# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# emotion_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
# emotion_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

# def get_emotion_embeddings(captions):
#     inputs = emotion_tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = emotion_model(**inputs)
#     return outputs.logits.numpy()


# # Assume emotion_embeddings and melody_embeddings are precomputed
# def build_emotion_to_melody_index(emotion_embeddings, melody_embeddings, M=32, efConstruction=128):
#     assert emotion_embeddings.shape[0] == melody_embeddings.shape[0], "Mismatch in data lengths!"
#     # Concatenate emotion and melody embeddings (or use mapping)
#     combined_data = np.concatenate((emotion_embeddings, melody_embeddings), axis=1)

#     # Build the HNSW index
#     index = faiss.IndexHNSWFlat(combined_data.shape[1], M)
#     index.hnsw.efConstruction = efConstruction
#     index.add(combined_data)
#     return index

# def query_emotion_to_melody(index, emotion_embedding, k=5):
#     _, melody_indices = index.search(emotion_embedding, k)
#     return melody_indices



# def build_hnsw_index(data, M=32, efConstruction=128):
#     """
#     Build HNSW index.

#     Args:
#         data (np.ndarray): Dataset with shape (n_samples, d).
#         M (int): Number of connections in HNSW.
#         efConstruction (int): ef parameter during construction.

#     Returns:
#         faiss.IndexHNSWFlat: Built HNSW index.
#     """
#     d = data.shape[1]
#     index = faiss.IndexHNSWFlat(d, M)
#     index.hnsw.efConstruction = efConstruction

#     start_time = time.time()
#     index.add(data)
#     end_time = time.time()

#     print(f"HNSW index construction time: {end_time - start_time:.2f} seconds")
#     return index

# def evaluate_index(audio_index, melody_index, audio_queries, k=1):
#     """
#     Evaluate index performance.

#     Args:
#         audio_index (faiss.Index): Audio index.
#         melody_index (faiss.Index): Melody index.
#         audio_queries (np.ndarray): Query vectors with shape (n_queries, d).
#         k (int): Number of nearest neighbors.

#     Returns:
#         dict: Evaluation results with search times.
#     """
#     # Evaluate audio to audio query
#     audio_to_audio_start_time = time.time()
#     _, _ = audio_index.search(audio_queries, k)
#     audio_to_audio_search_time = time.time() - audio_to_audio_start_time

#     # Evaluate audio to melody query
#     audio_to_melody_start_time = time.time()
#     _, _ = melody_index.search(audio_queries, k)
#     audio_to_melody_search_time = time.time() - audio_to_melody_start_time

#     return {
#         'k': k,
#         'audio_to_audio_search_time': audio_to_audio_search_time,
#         'audio_to_melody_search_time': audio_to_melody_search_time,
#     }

# def save_index(index, index_path):
#     """
#     Save FAISS index to specified path.

#     Args:
#         index (faiss.Index): Index to save.
#         index_path (str): File path to save the index.
#     """
#     directory = os.path.dirname(index_path)
#     os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
#     faiss.write_index(index, index_path)
#     print(f"Index saved to: {index_path}")

# if __name__ == "__main__":
#     # Load data
#     melody_data = np.load('/root/m2music/data/demo_embedding/melody_embeddings.npy')
#     print("melody_data loaded")
#     audio_data = np.load('/root/m2music/data/demo_embedding/audio_embeddings.npy')
#     print("audio_data loaded")
#     audio_queries = np.load('/root/m2music/data/demo_embedding/text_embeddings.npy')
#     print("audio_queries loaded")
    
#     # HNSW parameters
#     M = 32
#     efConstruction = 80
#     k = 1

#     print("Building melody HNSW index...")
#     melody_index = build_hnsw_index(melody_data, M=M, efConstruction=efConstruction)
    
#     print("Building audio HNSW index...")
#     audio_index = build_hnsw_index(audio_data, M=M, efConstruction=efConstruction)

#     index_info = f"HNSW, M: {M}, efConstruction: {efConstruction}"
#     file_suffix = "hnsw"

#     # Run validation
#     print(f"Running validation with k={k}...")
#     validation_result = evaluate_index(audio_index, melody_index, audio_queries, k=k)

#     # Output validation results
#     print(f"\nIndex type: {index_info}, k: {k}")
#     print(f"Audio to audio search time: {validation_result['audio_to_audio_search_time']:.6f} seconds")
#     print(f"Audio to melody search time: {validation_result['audio_to_melody_search_time']:.6f} seconds")

#     # Save index
#     save_path = '/root/m2music/data/faiss'

#     melody_index_path = os.path.join(save_path, f'audio_2_melody_{file_suffix}.faiss')
#     print("Saving melody HNSW index...")
#     save_index(melody_index, melody_index_path)

#     audio_index_path = os.path.join(save_path, f'audio_2_audio_{file_suffix}.faiss')
#     print("Saving audio HNSW index...")
#     save_index(audio_index, audio_index_path)

import numpy as np
import faiss
import time
import os

# Build HNSW index
def build_hnsw_index(data, M=32, efConstruction=128):
    d = data.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction

    start_time = time.time()
    index.add(data)
    end_time = time.time()

    print(f"HNSW index construction time: {end_time - start_time:.2f} seconds")
    return index

# Build emotion-to-melody HNSW index
def build_emotion_to_melody_index(emotion_embeddings, melody_embeddings, M=32, efConstruction=128):
    assert emotion_embeddings.shape[0] == melody_embeddings.shape[0], "Mismatch in data lengths!"
    print(f"Emotion embeddings shape:::::::::::::::::::::::::::::::::::::: {emotion_embeddings.shape}")  # Expected: (N, 1280)
    print(f"Melody embeddings shape::::::::::::::::::::::::::::::::::::::: {melody_embeddings.shape}")    # Expected: (N, 512)
    combined_data = np.concatenate((emotion_embeddings, melody_embeddings), axis=1)
    print(f"Combined data shape::::::::::::::::::::::::::::::::::::::::::: {combined_data.shape}")            # Expected: (N, 1792
    return build_hnsw_index(combined_data, M=M, efConstruction=efConstruction)

# Evaluate indexes
def evaluate_index(audio_index, melody_index, audio_queries, emotion_queries, k=1):
    audio_to_audio_start_time = time.time()
    _, _ = audio_index.search(audio_queries, k)  # Validate audio queries
    audio_to_audio_search_time = time.time() - audio_to_audio_start_time

    emotion_to_melody_start_time = time.time()
    combined_queries = np.concatenate((emotion_queries, audio_queries), axis=1)
    print(f"combined Query shape::::::::::::::::::::::::::::::::::::::::::::::: {combined_queries.shape}")  # Expected: (batch_size, 1792)
    _, _ = melody_index.search(combined_queries, k)
    emotion_to_melody_search_time = time.time() - emotion_to_melody_start_time

    return {
        'k': k,
        'audio_to_audio_search_time': audio_to_audio_search_time,
        'emotion_to_melody_search_time': emotion_to_melody_search_time,
    }

# Save FAISS index
def save_index(index, index_path):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"Index saved to: {index_path}")

if __name__ == "__main__":
    # Load data
    melody_data = np.load('/root/mg2emotion/data/melody_embeddings.npy')
    print("melody_data loaded")
    audio_data = np.load('/root/mg2emotion/data/audio_embeddings.npy')
    print("audio_data loaded")
    audio_queries = np.load('/root/mg2emotion/data/text_embeddings.npy')
    print("audio_queries loaded")
    emotion_data = np.load('/root/mg2emotion/data/emotion_embeddings.npy')
    print("emotion_data loaded")

    # HNSW parameters
    M = 32
    efConstruction = 80
    k = 1

    # Build indexes
    print("Building emotion-to-melody HNSW index...")
    emotion_to_melody_index = build_emotion_to_melody_index(emotion_data, melody_data, M=M, efConstruction=efConstruction)
    print("emotion_to_melody_index built::::::::::::::::::::::::::::::::::::::::::::::::::{emotion_to_melody_index.d}")
    print("Building audio HNSW index...")
    audio_index = build_hnsw_index(audio_data, M=M, efConstruction=efConstruction)

    # Validation
    print(f"Running validation with k={k}...")
    validation_result = evaluate_index(audio_index, emotion_to_melody_index, audio_queries, emotion_data, k=k)

    # Output validation results
    print(f"\nValidation Results:")
    print(f"Audio to audio search time: {validation_result['audio_to_audio_search_time']:.6f} seconds")
    print(f"Emotion to melody search time: {validation_result['emotion_to_melody_search_time']:.6f} seconds")

    # Save indexes
    save_path = '/root/mg2emotion/data/faiss'

    emotion_to_melody_index_path = os.path.join(save_path, 'emotion_to_melody_hnsw.faiss')
    print("Saving emotion-to-melody HNSW index...")
    save_index(emotion_to_melody_index, emotion_to_melody_index_path)

    audio_index_path = os.path.join(save_path, 'audio_hnsw.faiss')
    print("Saving audio HNSW index...")
    save_index(audio_index, audio_index_path)


