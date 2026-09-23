from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def save_embeddings(path: Path, matrix: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def load_embeddings(path: Path) -> Any:
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


def embed_query_and_candidates(
    embedding_model: str,
    query_text: str,
    candidate_texts: Sequence[str],
    batch_size: int,
    model_instance: Any = None,
    sample_dir: Path | None = None,
) -> Dict[str, Any]:
    """Embed one MS query and its independent structure candidate pool."""

    query_embedding_file: Path | None = None
    candidate_embeddings_file: Path | None = None
    if sample_dir is not None:
        sample_dir = Path(sample_dir)
        query_embedding_file = sample_dir / "query_embedding.npy"
        candidate_embeddings_file = sample_dir / "candidate_embeddings.npy"

    if (
        query_embedding_file is not None
        and candidate_embeddings_file is not None
        and query_embedding_file.is_file()
        and candidate_embeddings_file.is_file()
    ):
        query_embedding = load_embeddings(query_embedding_file)
        candidate_embeddings = load_embeddings(candidate_embeddings_file)
        return {
            "query_embedding": query_embedding,
            "candidate_embeddings": candidate_embeddings,
        }

    if embedding_model == "mock-random":
        # Reproducible random vectors for plumbing tests only. These are not metrics.
        import hashlib
        import numpy as np

        def vector_for(text: str) -> Any:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            vector = np.random.default_rng(seed).standard_normal(128).astype("float32")
            norm = np.linalg.norm(vector)
            return vector / (norm if norm else 1.0)

        query_embedding = np.stack([vector_for(query_text)])
        candidate_embeddings = np.stack([vector_for(text) for text in candidate_texts])
    elif embedding_model == "tfidf-char-ngram":
        # Explicitly an offline smoke-test backend, not a benchmark model.
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        matrix = vectorizer.fit_transform([query_text, *candidate_texts]).astype("float32")
        query_embedding = matrix[:1].toarray()
        candidate_embeddings = matrix[1:].toarray()
    else:
        model = model_instance or load_sentence_transformer(embedding_model)
        query_embedding = model.encode(
            [query_text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        candidate_embeddings = model.encode(
            list(candidate_texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    if query_embedding_file is not None and candidate_embeddings_file is not None:
        save_embeddings(query_embedding_file, query_embedding)
        save_embeddings(candidate_embeddings_file, candidate_embeddings)

    return {
        "query_embedding": query_embedding,
        "candidate_embeddings": candidate_embeddings,
    }
