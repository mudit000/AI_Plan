# Day 2 — Tokenization & Embeddings

![Tokenization to embedding pipeline](./img/day2-diagram.svg)
 
*Raw text gets split into sub-word tokens, each token maps to an integer ID via a vocabulary lookup, and the full sequence collapses into one embedding vector representing the sentence's meaning. Today you'll generate every stage of this yourself.*

**Goal for today:** open up the two things that happen to your text *before* a model ever "thinks" about it — splitting into tokens, and converting into a meaning-vector. By the end you'll be able to explain, with your own generated numbers, why models can't count letters in "strawberry" and how semantic search actually works under the hood.

---

## Part A — Concepts (10 min read)

**Why tokenization exists:** neural networks only operate on numbers. Text has to become numbers somehow. The naive options both fail at scale:
- **Whole-word vocabulary** — every possible word gets an ID. Fails on typos, rare words, made-up words, other languages — infinite vocabulary problem.
- **Character-level** — every letter gets an ID. Works for any text, but sequences become very long (slow, eats context window fast).

**The solution — sub-word tokenization (BPE, Byte-Pair Encoding):** break text into common *chunks* — sometimes whole words, sometimes word pieces, sometimes single characters for rare stuff. Common words like "the" get their own token; rare/compound words like "kubectl" or "nginx" get split into pieces. This is a middle ground that keeps sequences short while still handling any input text.

**Why this explains a famous LLM failure mode:** the "strawberry" letter-counting problem happens because the model never sees individual letters — it sees 2-3 opaque token chunks for that word. Asking it to count letters is like asking you to count the pixels in a word you're reading — the information isn't wrong, it's just not the representation you're operating on.

**Context window, concretely:** every model has a max token count per request (input + output combined) — e.g. 8K, 32K, 128K tokens. This is a hard resource limit, same category as a max payload size on an API gateway. Exceed it and older content gets silently dropped or the request fails, depending on the client.

**Why tokens matter for cost/ops, not just theory:** hosted APIs (OpenAI, Anthropic, etc.) bill per token, not per word or per character. Token counting is the equivalent of estimating compute cost before running a batch job — you'll do this for real in Step 3.

**What an embedding is (recap + extension from the prereq doc):** a fixed-length vector of floats representing the *meaning* of a piece of text. Today's focus is generating and inspecting these directly, at the sentence level, independent of any vector DB — the DB is just where you *store and search* them later.

---

## Part B — Hands-On: Tokenization

### Step 1 — Encode and decode with tiktoken

Purpose: see exactly what a tokenizer does — text in, integers out, and back again.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # the tokenizer family used by GPT-3.5/4-era models

text = "restarting nginx after a failed health check"
tokens = enc.encode(text)

print("Token IDs:", tokens)
print("Token count:", len(tokens))

for t in tokens:
    print(f"{t:6d} -> {repr(enc.decode([t]))}")
```
```bash
(venv) muditcse@Mac ai-course % python3 Encode_and_decode_with_tiktoken.py 
Token IDs: [265, 40389, 71582, 1306, 264, 4745, 2890, 1817]
Token count: 8
   265 -> 're'
 40389 -> 'starting'
 71582 -> ' nginx'
  1306 -> ' after'
   264 -> ' a'
  4745 -> ' failed'
  2890 -> ' health'
  1817 -> ' check'
(venv) muditcse@Mac ai-course % 
```

**What to notice:** common words ("after", "a", "failed") often become single tokens, while less common ones may split into pieces. Run `enc.decode(tokens)` and confirm you get the exact original string back — tokenization is lossless, just a different representation.

```bash
(venv) muditcse@Mac ai-course % python3 Encode_and_decode_with_tiktoken.py
Token IDs: [265, 40389, 71582, 1306, 264, 4745, 2890, 1817]
Token count: 8
   265 -> 'restarting nginx after a failed health check'
 40389 -> 'restarting nginx after a failed health check'
 71582 -> 'restarting nginx after a failed health check'
  1306 -> 'restarting nginx after a failed health check'
   264 -> 'restarting nginx after a failed health check'
  4745 -> 'restarting nginx after a failed health check'
  2890 -> 'restarting nginx after a failed health check'
  1817 -> 'restarting nginx after a failed health check'
(venv) muditcse@Mac ai-course % 
```

### Step 2 — Prove the "strawberry problem" to yourself

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
for word in ["strawberry", "kubectl", "nginx", "the", "unbelievable"]:
    toks = enc.encode(word)
    pieces = [enc.decode([t]) for t in toks]
    print(f"{word:15s} -> {len(toks)} token(s): {pieces}")
```
```bash
(venv) muditcse@Mac ai-course % python3 strawberry_problem.py                
strawberry      -> 3 token(s): ['str', 'aw', 'berry']
kubectl         -> 1 token(s): ['kubectl']
nginx           -> 1 token(s): ['nginx']
the             -> 1 token(s): ['the']
unbelievable    -> 3 token(s): ['un', 'belie', 'vable']
(venv) muditcse@Mac ai-course % 
```

**What to expect:** `"the"` is 1 token, `"strawberry"` is likely 2-3 tokens, `"kubectl"` and `"nginx"` (less common in general text, common in your world) may split unexpectedly. This is why domain-specific jargon sometimes tokenizes inefficiently — the base tokenizer's vocabulary was built from general internet text, not your infra vocabulary.

### Step 3 — Practical use case: estimate request cost before sending it

Purpose: this is the real reason engineers check token counts — treat it like estimating the cost of a cloud job before submitting it.

```python
def estimate_cost(text, price_per_1k_tokens=0.0025):
    n_tokens = len(enc.encode(text))
    cost = (n_tokens / 1000) * price_per_1k_tokens
    return n_tokens, cost

runbook = open("docs/runbook.txt").read() if __import__("os").path.exists("docs/runbook.txt") else "Rotate the TLS cert 30 days before expiry."
n, cost = estimate_cost(runbook)
print(f"{n} tokens, ~${cost:.5f} at this rate")
```
Run this against something realistic — a full runbook, a long prompt template, or a big RAG context block — and you have a real pre-flight cost check, the same instinct as `terraform plan` before `apply`.
```bash

(venv) muditcse@Mac ai-course % python3 estimate_request_cost.py
10 tokens, ~$0.00003 at this rate
(venv) muditcse@Mac ai-course % mkdir docs
(venv) muditcse@Mac ai-course % touch docs/runbook.txt
(venv) muditcse@Mac ai-course % vi docs/runbook.txt
(venv) muditcse@Mac ai-course % python3 estimate_request_cost.py
78 tokens, ~$0.00019 at this rate
(venv) muditcse@Mac ai-course % 
```
### Step 4 — Compare tokenizers across model families

Purpose: different model families use different tokenizers, so the *same* text produces *different* token counts and context-window usage depending on which model you target. This matters when picking a model for a cost- or latency-sensitive workload.

```python
from transformers import AutoTokenizer

text = "The HPA will scale replicas when CPU exceeds 80 percent for five minutes."

for model_name in ["gpt2", "bert-base-uncased"]:
    tok = AutoTokenizer.from_pretrained(model_name)
    ids = tok.encode(text)
    print(f"{model_name:20s} -> {len(ids)} tokens")
```
**What to notice:** the token count differs between tokenizers for identical input text. This is why "128K context window" means different actual amounts of *your* text depending on which model's tokenizer is counting.
```bash
(venv) muditcse@Mac ai-course % python3 Compare_tokenizers_across_model_families.py 
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 665/665 [00:00<00:00, 1.37MB/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 53.6kB/s]
vocab.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:00<00:00, 4.62MB/s]
merges.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 4.63MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.36M/1.36M [00:00<00:00, 5.33MB/s]
gpt2                 -> 16 tokens
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 570/570 [00:00<00:00, 1.99MB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 145kB/s]
vocab.txt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 7.00MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 466k/466k [00:00<00:00, 3.33MB/s]
bert-base-uncased    -> 18 tokens
(venv) muditcse@Mac ai-course % 
```
---

## Part C — Hands-On: Embeddings

### Step 5 — Generate your first embedding and look at the raw vector

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("restarting nginx after a failed health check")

print("Shape:", vec.shape)     # (384,)
print("First 10 values:", vec[:10])
print("Min/max:", vec.min(), vec.max())
```
**What to notice:** 384 numbers, each typically small (roughly -1 to 1 range for this model). No single number means anything on its own — the pattern across all 384 encodes the sentence's meaning. This is the same object type a vector DB stores per document, just generated standalone here.

```bash
(venv) muditcse@Mac ai-course % python3 first_embedding_raw_vector.py 
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
README.md: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10.5k/10.5k [00:00<00:00, 21.9MB/s]
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6029.75it/s]
Shape: (384,)
First 10 values: [-0.00473463  0.05080811 -0.04674405 -0.0026344   0.01762828 -0.04631952
  0.00402148 -0.02703921 -0.02799736 -0.00091216]
Min/max: -0.13277146 0.16864385
(venv) muditcse@Mac ai-course % 
```
### Step 6 — Batch-embed a realistic set of devops sentences

```python
sentences = [
    "Restart the nginx service if health checks fail.",
    "Bounce nginx after repeated probe failures.",
    "Scale up pod replicas when CPU usage is high.",
    "Increase the number of replicas under heavy load.",
    "Rotate the TLS certificate before it expires.",
    "Renew the cert-manager certificate ahead of expiry.",
    "Deploy the new build to the staging environment.",
]

vecs = model.encode(sentences)
print(vecs.shape)   # (7, 384) - 7 sentences, 384 dims each
```

```bash
(venv) muditcse@Mac ai-course % python3 batch_embedded.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 10190.92it/s]
(7, 384)
(venv) muditcse@Mac ai-course % 
```
### Step 7 — Compute a full similarity matrix and read it like a heatmap table

Purpose: this is the computational core of every "semantic search" feature you'll ever build — seeing the whole matrix at once shows *why* it works, not just one query at a time.

```python
from sentence_transformers import util
import numpy as np

sim_matrix = util.cos_sim(vecs, vecs).numpy()

print("     " + " ".join(f"S{i}" for i in range(len(sentences))))
for i, row in enumerate(sim_matrix):
    print(f"S{i}: " + " ".join(f"{v:.2f}" for v in row))
```
**What to notice:** sentences 0-1 (both about nginx restarts) should show high similarity to each other and lower similarity to sentences 4-5 (about certs). The diagonal is always 1.00 (a sentence is identical to itself). This matrix, at scale, is exactly what a vector DB's index is built to search efficiently instead of computing brute-force like this.

```bash
(venv) muditcse@Mac ai-course % python3 similarity_matrix_heatmap_table.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 2898.23it/s]
     S0 S1 S2 S3 S4 S5 S6
S0: 1.00 0.59 0.08 0.12 0.21 0.18 0.12
S1: 0.59 1.00 0.09 0.12 0.16 0.09 0.08
S2: 0.08 0.09 1.00 0.63 -0.00 0.00 0.11
S3: 0.12 0.12 0.63 1.00 0.15 0.14 0.29
S4: 0.21 0.16 -0.00 0.15 1.00 0.74 0.20
S5: 0.18 0.09 0.00 0.14 0.74 1.00 0.20
S6: 0.12 0.08 0.11 0.29 0.20 0.20 1.00
(venv) muditcse@Mac ai-course % 
```
### Step 8 — Visualize the clustering (see it, don't just read numbers)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

coords = PCA(n_components=2).fit_transform(vecs)
topics = ["nginx", "nginx", "scale", "scale", "cert", "cert", "deploy"]
colors = {"nginx": "tab:green", "scale": "tab:blue", "cert": "tab:red", "deploy": "tab:orange"}

plt.figure(figsize=(6, 5))
for (x, y), topic, s in zip(coords, topics, sentences):
    plt.scatter(x, y, c=colors[topic], s=90)
    plt.annotate(s[:25], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
plt.title("DevOps sentences clustered by meaning")
plt.savefig("day2_embeddings.png", dpi=150, bbox_inches="tight")
print("Saved day2_embeddings.png")
```
Open the PNG — you should see nginx-related, scaling-related, and cert-related sentences forming visibly separate clusters, with "deploy" sitting alone since nothing else relates to it.
```bash
```
### Step 9 — Practical use case: near-duplicate detection (a real ops problem)

Purpose: this is a genuine production use case — flagging near-duplicate incident tickets, log lines, or alerts even when the wording differs, so you don't get paged three times for the same underlying issue phrased differently.

```python
def find_near_duplicates(sentences, threshold=0.75):
    vecs = model.encode(sentences)
    sims = util.cos_sim(vecs, vecs).numpy()
    pairs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if sims[i][j] > threshold:
                pairs.append((sentences[i], sentences[j], sims[i][j]))
    return pairs

tickets = [
    "nginx pod keeps restarting due to failed liveness probe",
    "Liveness probe failing, nginx container restart loop",
    "Database connection pool exhausted under load",
    "TLS cert expired on the ingress controller",
]

for a, b, score in find_near_duplicates(tickets):
    print(f"[{score:.2f}] '{a}'  ~~  '{b}'")
```
**Expected result:** the two nginx/liveness-probe tickets should pair up as near-duplicates despite completely different wording — this is a directly deployable pattern for ticket dedup or alert grouping.
```bash
```
---

## Deliverable

Save to `~/ai-course/day2-notes.md`:

```markdown
# Day 2 Notes

## Tokenization
- "strawberry" tokenized into: ...
- Token count difference between gpt2 and bert tokenizers for the same sentence: ...
- Estimated cost of my runbook.txt: ...

## Embeddings
- Similarity score between the two nginx sentences: ...
- Similarity score between an nginx sentence and a cert sentence: ...
- Near-duplicate pairs found: ...

## One thing that surprised me:
...
```

---
