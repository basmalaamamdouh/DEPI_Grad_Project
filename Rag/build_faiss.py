import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# 1. Load chunks
# =========================
with open("chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

print(f"Loaded {len(texts)} chunks")


# =========================
# 2. Load embedding model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 3. Create embeddings
# =========================
print("Creating embeddings...")

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# =========================
# 4. Build FAISS index
# =========================
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index created with {index.ntotal} vectors")


# =========================
# 5. Save index + metadata
# =========================
faiss.write_index(index, "cv_index.faiss")

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Saved FAISS index + metadata")