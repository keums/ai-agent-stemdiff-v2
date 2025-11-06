# modules/embedding_ranker.py

import os
import pathlib
from typing import Dict, List

import numpy as np
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_embedding(text: str) -> List[float]:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


EMBEDDING_SUFFIX = ".embedding.npz"


def _build_text_section(meta: Dict) -> str:
    user = meta.get("user_prompt", "") or ""
    intent = meta.get("intent_prompt", "") or ""
    return f"[USER_PROMPT]\n{user}\n\n[INTENT_PROMPT]\n{intent}"


def _build_stems_section(meta: Dict) -> str:
    def to_pairs(stems: List[Dict]) -> List[str]:
        pairs: List[str] = []
        for s in stems or []:
            category = s.get("category", "") or ""
            caption = s.get("caption", "") or ""
            pairs.append(f"{category}: {caption}")
        return pairs

    mix_pairs = to_pairs(meta.get("mix_stems", []))
    suggested_pairs = to_pairs(meta.get("suggested_stems", []))
    mix_str = " \n".join(mix_pairs)
    suggested_str = " \n".join(suggested_pairs)
    return f"[MIX_STEMS]\n{mix_str}\n\n[SUGGESTED_STEMS]\n{suggested_str}"


def _build_combined_section(meta: Dict) -> str:
    text_str = _build_text_section(meta)
    stems_str = _build_stems_section(meta)
    return f"{text_str}\n\n{stems_str}"


def get_or_create_embedding_for_memory(meta: Dict) -> List[float]:
    """Return a single embedding vector. Cache key is 'embedding' (with legacy support)."""
    emb_path = meta["file_path"].replace(".json", EMBEDDING_SUFFIX)

    if os.path.exists(emb_path):
        loaded = np.load(emb_path)
        return loaded["embedding"].astype(np.float32).tolist()

    combined_str = _build_combined_section(meta)
    vec = get_embedding(combined_str)
    np.savez_compressed(emb_path, embedding=np.array(vec, dtype=np.float16))
    return vec


def _token_overlap(query: str, text: str) -> float:
    q = set((query or "").lower().split())
    hay = (text or "").lower()
    if not q:
        return 0.0
    return sum(1.0 for token in q if token in hay) / len(q)


def lexical_section_scores(query: str, meta: Dict) -> Dict[str, float]:
    """Compute lightweight lexical overlap for user+intent only."""
    text_section = _build_text_section(meta)
    return {"lex_text": _token_overlap(query, text_section)}


def get_embedding_ranked_candidates(
    user_prompt: str, candidates: List[dict], top_k=5
) -> List[dict]:
    user_vec = get_embedding(user_prompt)
    ranked = []

    for meta in candidates:
        memory_vec = get_or_create_embedding_for_memory(meta)
        emb_score = (
            cosine_similarity(user_vec, memory_vec) if len(memory_vec) > 0 else 0.0
        )
        lex_score = lexical_section_scores(user_prompt, meta)["lex_text"]
        score = 0.9 * emb_score + 0.1 * lex_score
        ranked.append((score, meta))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return [r[1] for r in ranked[:top_k]]


# def get_or_create_embedding_for_memory(meta: dict) -> List[float]:
#     emb_path = meta["file_path"].replace(".json", ".embedding.json")
#     if os.path.exists(emb_path):
#         with open(emb_path, "r") as f:
#             return json.load(f)["embedding"]

#     captions = " ".join(
#         s.get("caption", "") for s in meta["mix_stems"] + meta["suggested_stems"]
#     )
#     context_text = meta["user_prompt"] + " " + meta["intent_prompt"] + " " + captions
#     emb = get_embedding(context_text)
#     with open(emb_path, "w") as f:
#         compressed = np.array(emb, dtype=np.float16).tolist()
#         json.dump({"embedding": compressed}, f)
#     return emb


# def get_embedding_ranked_candidates(
#     user_prompt: str, candidates: List[dict], top_k=5
# ) -> List[dict]:
#     user_vec = get_embedding(user_prompt)
#     ranked = []

#     for meta in candidates:
#         memory_vec = get_or_create_embedding_for_memory(meta)
#         score = cosine_similarity(user_vec, memory_vec)
#         ranked.append((score, meta))

#     ranked.sort(reverse=True, key=lambda x: x[0])
#     return [r[1] for r in ranked[:top_k]]
