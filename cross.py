from sentence_transformers import CrossEncoder
for name, d in [("cross-encoder/ms-marco-MiniLM-L-6-v2","minilm-l6"), ("BAAI/bge-reranker-base","bge-base")]:
    CrossEncoder(name).save(f"rag/models/rerankers/{d}")
