import torch

import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from concept_erasure import LeaceEraser # https://github.com/EleutherAI/concept-erasure/

def purity_score(z):
    p = (z == 1).float().mean().item()
    return round(max(p, 1 - p), 2)

model = SentenceTransformer("all-mpnet-base-v2")
data = pd.read_json("data/bills_tweets_sample.jsonl", lines=True)

X = model.encode(
    data["text"].tolist(),
    show_progress_bar=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    convert_to_tensor=True,
).cpu()
y = torch.tensor(data["source"] == "tweet").long()

# without concept erasure
k = 5
kmeans = KMeans(n_clusters=k, random_state=11235)
kmeans.fit(X.numpy())
# report the purity of each cluster
print(sorted([purity_score(y[kmeans.labels_ == i]) for i in range(k)], reverse=True))
#> [1.0, 0.99, 0.98, 0.98, 0.98]

# with concept erasure
eraser = LeaceEraser.fit(X, y)
X_erased = eraser(X)

kmeans_erased = KMeans(n_clusters=k, random_state=11235)
kmeans_erased.fit(X_erased.numpy())

print(sorted([purity_score(y[kmeans_erased.labels_ == i]) for i in range(k)],))
#> [0.59, 0.59, 0.55, 0.54, 0.51]
