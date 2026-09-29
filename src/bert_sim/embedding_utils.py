from typing import Any, Dict, List, Sequence


def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def save_embeddings(path, matrix: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def load_embeddings(path) -> Any:
    import numpy as np
    data = np.load(path, allow_pickle=False)
    return data


def cosine_similarity_matrix(embeddings: Any) -> Any:
    import numpy as np

    return np.matmul(embeddings, embeddings.T)


def rank_neighbors(
    record_ids: Sequence[str],
    similarity_matrix: Any,
    top_k: int,
) -> List[Dict[str, Any]]:
    import numpy as np

    neighbors: List[Dict[str, Any]] = []
    for idx, record_id in enumerate(record_ids):
        row = similarity_matrix[idx].copy()
        row[idx] = -1.0
        top_indices = np.argsort(-row)[:top_k]
        neighbors.append(
            {
                "record_id": record_id,
                "neighbors": [
                    {"record_id": record_ids[j], "score": float(row[j])}
                    for j in top_indices
                ],
            }
        )
    return neighbors


def embed_texts(
    embedding_model: str,
    ms_texts: Sequence[str],
    structure_texts: Sequence[str],
    combined_texts: Sequence[str],
    batch_size: int,
) -> Dict[str, Any]:
    model = load_sentence_transformer(embedding_model)
    ms_embeddings = model.encode(
        list(ms_texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    structure_embeddings = model.encode(
        list(structure_texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    combined_embeddings = model.encode(
        list(combined_texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {
        "ms_embeddings": ms_embeddings,
        "structure_embeddings": structure_embeddings,
        "combined_embeddings": combined_embeddings,
    }

