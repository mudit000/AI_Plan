# Prereq: Registry / Vector DB / RAG — DevOps Analogies + Local Hands-On

Do this before Day 8-9 of the main curriculum. Goal: build intuition for three core concepts by mapping them to infra patterns you already know, then run each one locally.

---

## 1. Model Registry ≈ Artifact Repo (Nexus/Artifactory/ECR)

**The mapping:** A model registry stores versioned model *artifacts* (weights + config + metadata) the same way Artifactory stores versioned jars or ECR stores image tags. You pull a specific version, not "latest" blindly, because behavior changes between versions just like a new image build can break things.

**What's actually in a "model artifact":**
- Weights file (`.safetensors`, `.gguf`, `.bin`)
- Config (architecture, context length, tokenizer)
- Metadata (license, training data card, version tag)

### Hands-on

```bash
# Ollama IS a local model registry + runtime combined
ollama list                  # like `docker images`
ollama show llama3.2         # like `docker inspect` - shows the Modelfile, params, template
ollama pull mistral:7b       # pulling a specific "tag"

# Inspect the actual artifact on disk
ls ~/.ollama/models/manifests/registry.ollama.ai/library/
cat ~/.ollama/models/manifests/registry.ollama.ai/library/llama3.2/latest
```

This manifest file is a JSON pointer to content-addressed blobs — same pattern as Docker image layers (SHA-addressed, deduplicated). Not a coincidence: Ollama borrowed the OCI registry format on purpose.

**Task — create a custom "version" like tagging a custom image:**

```bash
cat > Modelfile <<'EOF'
FROM llama3.2
PARAMETER temperature 0.3
SYSTEM "You are a terse DevOps assistant. Answer in bullet points only."
EOF

ollama create devops-bot:v1 -f Modelfile
ollama run devops-bot:v1
```

`ollama list` now shows `devops-bot:v1` as its own versioned artifact — a registry push/tag workflow.

### Optional — the "real" registry experience (Hugging Face Hub)

Ollama hides the registry mechanics behind a nice CLI. Hugging Face Hub is the raw artifact-repo experience — closer to Artifactory/Nexus, with repos, branches, commits (SHAs), and file-level pulls. Worth doing once so you see what Ollama is abstracting away.

**a) Setup + auth (like `docker login` / configuring a Nexus credential):**
```bash
pip install -U huggingface_hub
huggingface-cli login          # paste a token from huggingface.co/settings/tokens
huggingface-cli whoami
```

**b) Browse a repo like you'd browse an Artifactory path:**
```bash
# List every file in a model repo before pulling anything - like `curl` listing artifact paths
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf --dry-run 2>/dev/null || \
python3 -c "
from huggingface_hub import list_repo_files
for f in list_repo_files('microsoft/Phi-3-mini-4k-instruct-gguf'):
    print(f)
"
```
You'll see multiple quantization variants (`Phi-3-mini-4k-instruct-q4.gguf`, `-fp16.gguf`, etc.) sitting side by side in the same repo — like multiple build artifacts (debug/release/arm64) published under one project.

**c) Pull just one file, not the whole repo (like grabbing a single artifact instead of the whole build):**
```bash
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='microsoft/Phi-3-mini-4k-instruct-gguf',
    filename='Phi-3-mini-4k-instruct-q4.gguf'
)
print('Downloaded to:', path)
"
```

**d) Pull a full repo snapshot at a pinned revision (like pinning a git SHA or an image digest instead of `latest`):**
```bash
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='microsoft/Phi-3-mini-4k-instruct-gguf',
    revision='main'          # swap for a commit SHA to pin an exact version
)
print(path)
"
```
Every HF repo is a git repo under the hood — `revision` accepts a branch name, a tag, or a raw commit SHA. This is the direct equivalent of pinning `image@sha256:...` instead of trusting `:latest`.

**e) Inspect metadata the way you'd inspect an artifact's manifest:**
```bash
python3 -c "
from huggingface_hub import model_info
info = model_info('microsoft/Phi-3-mini-4k-instruct-gguf')
print('SHA:', info.sha)
print('Last modified:', info.lastModified)
print('Tags:', info.tags)
print('Siblings (files):', [s.rfilename for s in info.siblings])
"
```

**f) See where it actually lives on disk (the local cache, like `/var/lib/docker` or Artifactory's blob store):**
```bash
ls -la ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/*
```
Notice the folder naming: `models--<org>--<repo>` with a `blobs/` dir (content-addressed by hash) and `snapshots/<commit-sha>/` (symlinks into blobs). Same content-addressable-storage pattern as Docker layers and Ollama's manifest/blob split you looked at earlier — three different tools, same underlying design because it solves the same problem (dedupe + immutability + fast diffing).

**g) Compare two revisions (like diffing two artifact versions):**
```bash
python3 -c "
from huggingface_hub import list_repo_commits
commits = list_repo_commits('microsoft/Phi-3-mini-4k-instruct-gguf')
for c in commits[:5]:
    print(c.commit_id[:8], c.created_at, c.title)
"
```
This is literally `git log` for a model — each commit is a new version of the artifact, same idea as a new build number.

---

## 2. Vector DB ≈ Indexed Cache (a search index, not a KV store)

**Why it's required — the problem in one sentence:** keyword search (grep, SQL `LIKE`, Elasticsearch's default mode) only finds things that share *words*. It cannot find things that share *meaning*. A vector DB solves exactly that gap, and nothing else — that's its whole purpose. Everything below builds toward *seeing* that gap and then closing it.

**The core picture to hold in your head:** every piece of text gets converted into a list of numbers (a vector) that represents its *meaning* as a point in space. Texts with similar meaning land near each other in that space, regardless of which words they used. A vector DB is just a data store built to answer one question fast at scale: *"which points are nearest to this point?"*

### Step 1 — See the problem first (before any tooling)

Purpose: feel the limitation of keyword search with your own hands, so the rest of this makes sense as a solution rather than a buzzword.

```bash
mkdir -p ~/vectordb-lab && cd ~/vectordb-lab
echo "Rotate the TLS cert 30 days before expiry using cert-manager." > runbook.txt

grep -i "expiring certificate" runbook.txt   # returns nothing
```
**What just happened:** the runbook is *about* an expiring certificate, but the word "expiring" never appears — it says "before expiry". Grep is a string matcher, it has no concept that "expiring" and "expiry" mean the same thing, or that "cert" and "certificate" are the same concept. This is the exact gap a vector DB exists to close. Use case in the real world: anyone searching a knowledge base, support ticket system, or runbook set in their own words instead of the exact original phrasing.

### Step 2 — Turn text into numbers (embeddings) and *look at them*

Purpose: an embedding isn't magic — it's just a list of floats. Seeing the raw vector once demystifies everything downstream.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, runs fine on CPU/Mac
vec = model.encode("Rotate the TLS cert before it expires")

print(type(vec))       # numpy array
print(vec.shape)       # (384,) - 384 numbers represent this entire sentence
print(vec[:10])        # peek at the first 10 numbers - just floats, nothing exotic
```
**What to notice:** a whole sentence collapsed into 384 numbers. Those numbers have no human-readable meaning individually — the *pattern* across all 384 is what encodes "this sentence is about certificate rotation." This is the atomic unit a vector DB stores and searches over.

### Step 3 — Prove that distance between vectors = distance in meaning

Purpose: this is the mechanism, made visible with numbers before you trust a database to do it for you.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Rotate the TLS cert before it expires",      # A
    "The certificate is about to expire",         # B - same meaning, zero shared key words
    "Scale the pods when CPU usage is high",       # C - unrelated meaning
]
vecs = model.encode(sentences)

sim_AB = util.cos_sim(vecs[0], vecs[1]).item()
sim_AC = util.cos_sim(vecs[0], vecs[2]).item()

print(f"A vs B (both about certs):     {sim_AB:.3f}")
print(f"A vs C (cert vs scaling):      {sim_AC:.3f}")
```
**What to expect and why it matters:** A-vs-B scores high (roughly 0.6-0.8+) despite sharing almost no words. A-vs-C scores much lower. This one comparison *is* the entire value proposition of semantic search — cosine similarity between vectors tracks meaning, not spelling.

### Step 4 — See it visually: plot the vectors in 2D

Purpose: 384 dimensions is impossible to picture, but if you compress it down to 2D you can literally watch related sentences cluster together — the same picture as the diagram above, generated from your own data.

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "Rotate the TLS cert before it expires",
    "The certificate is about to expire",
    "cert-manager handles TLS renewal",
    "Scale the pods when CPU usage is high",
    "Increase replicas under heavy load",
    "Restart nginx if health checks fail",
    "Bounce the service after repeated failures",
]
labels = ["cert", "cert", "cert", "scale", "scale", "restart", "restart"]

vecs = model.encode(docs)
coords = PCA(n_components=2).fit_transform(vecs)   # compress 384 dims -> 2 dims for plotting

colors = {"cert": "tab:red", "scale": "tab:blue", "restart": "tab:green"}
plt.figure(figsize=(6, 5))
for (x, y), label, doc in zip(coords, labels, docs):
    plt.scatter(x, y, c=colors[label], s=80)
    plt.annotate(doc[:20], (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
plt.title("Sentences plotted by meaning, not by words")
plt.savefig("embedding_plot.png", dpi=150, bbox_inches="tight")
print("Saved embedding_plot.png - open it in Finder")
```
Run it, then `open embedding_plot.png`. **What you'll see:** the three "cert" sentences cluster together, the two "scale" sentences cluster together, the two "restart" sentences cluster together — despite different wording within each group. This is the exact picture a vector DB uses internally, just at a scale of millions of points instead of seven.

### Step 5 — Now bring in the actual database (ChromaDB)

Purpose: steps 1-4 proved the *concept* works with raw math. A vector DB exists because computing distance against millions of vectors one-by-one (like your 3-line loop above) doesn't scale — you need an index. This step shows the database doing what you just did by hand, but production-ready.

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")  # like a local data dir, ~/var/lib/db
ef = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection("runbooks", embedding_function=ef)

collection.add(
    documents=[
        "Restart the nginx service if health checks fail three times.",
        "Rotate the TLS cert 30 days before expiry using cert-manager.",
        "Scale the pod replicas when CPU exceeds 80% for 5 minutes.",
    ],
    ids=["doc1", "doc2", "doc3"]
)

results = collection.query(query_texts=["what do I do about an expiring certificate"], n_results=2)
print(results["documents"])
print(results["distances"])  # lower distance = closer meaning = better match
```
Run this — the query never says "TLS" or "cert-manager" verbatim, yet it retrieves doc2, and you can see its distance score is the lowest. Compare directly to Step 1's grep failure on the same query — that's the before/after in one pair of commands.

### Step 6 — Understand *why* a real index beats brute-force search (the part that makes it a "database" and not just a script)

Purpose: this is the piece that maps to your existing DB-indexing knowledge and explains what you're actually paying for when you pick Chroma/Pinecone/Weaviate/pgvector over a for-loop.

- **Brute-force (what Steps 2-4 did):** compare the query vector to *every single* stored vector, one by one. Fine for 10 docs, falls over at 10 million (O(n) per query).
- **What a vector DB adds:** an approximate nearest-neighbor index (commonly **HNSW** — Hierarchical Navigable Small World graph) built at insert time, so a query only touches a small fraction of vectors instead of all of them (roughly O(log n)). This is directly analogous to why you'd never `SELECT * FROM table WHERE ...` scan a billion-row table without a B-tree index — same tradeoff, different math.
- **The tradeoff to know about (like index tuning you already do):** it's called *approximate* nearest neighbor for a reason — HNSW trades a small amount of recall accuracy for large speed gains. That's a tunable knob (`ef_search`, `M` in HNSW terms), same category of decision as index fill-factor or shard count.

```python
# See it under the hood - Chroma exposes the same distance/index concepts
print(collection.count())                 # how many vectors are indexed
print(collection.peek(limit=1))           # raw stored record: id, embedding, document, metadata
```

### Purpose recap — why each piece exists

| Concept | DevOps analogy | Why it's needed |
|---|---|---|
| Embedding | Serializing an object to a fixed-size fingerprint | Turns unstructured text into something math can compare |
| Cosine similarity | Diffing two config files for "how similar" | The actual "closeness = relatedness" measurement |
| Vector index (HNSW) | B-tree / inverted index | Makes similarity search fast at scale, not O(n) per query |
| Vector DB (Chroma etc.) | The database wrapping the index | Persistence, filtering by metadata, CRUD, concurrent access |

### Use cases this unlocks (why you'd actually reach for this at work)

- **RAG** (next section) — retrieve relevant internal docs to ground an LLM's answer
- **Semantic search over runbooks/tickets/wikis** — find answers by intent, not exact phrasing
- **Deduplication** — flag near-duplicate incident tickets or log lines even when worded differently
- **Anomaly/outlier detection** — a log line whose embedding sits far from every known cluster is novel/unusual
- **Recommendation** — "similar to this" for docs, code snippets, or alerts

### Task — compare to grep to feel the difference one more time, end to end

```bash
echo "Restart the nginx service..." > runbook.txt
grep -i "expiring certificate" runbook.txt   # returns nothing - no exact match
```

Grep fails, vector search succeeds. That gap, made visible in Steps 1 and 4-5, is the entire reason vector DBs exist.

---

## 3. RAG ≈ Sidecar That Injects Context

**Why it's required — the problem in one sentence:** an LLM only knows what was in its training data, frozen at a cutoff date, and it has zero knowledge of your private docs, runbooks, or recent changes. RAG's entire purpose is to hand the model the *right* private/current information at request time, without retraining it — same reason a sidecar injects config/secrets into a request instead of rebuilding the main container's image.

**The mapping:** like a sidecar container that intercepts a request, enriches it (adds auth headers, injects config from a mounted secret), then passes it along — RAG intercepts your prompt, fetches relevant context from the vector DB, and injects it into the prompt *before* it hits the model. The model itself is never modified.

Everything below builds the pipeline in the diagram above, one box at a time, using the ChromaDB collection you already built in Section 2.

### Step 1 — Build a small but real document set

Purpose: RAG only proves itself on content the base model *couldn't* already know. Use your own text, not textbook examples.

```bash
mkdir -p ~/rag-lab/docs && cd ~/rag-lab
cat > docs/runbook.txt <<'EOF'
Certificate rotation: cert-manager auto-renews TLS certs 30 days before expiry.
If auto-renewal fails, manually run: kubectl cert-manager renew <cert-name>.

Nginx health checks: restart the nginx service if health checks fail three
consecutive times. Check logs at /var/log/nginx/error.log first.

Autoscaling policy: scale pod replicas when CPU exceeds 80% for 5 minutes.
Max replicas is capped at 12 per the current HPA config.
EOF
```

### Step 2 — Chunking: split documents into retrievable units

Purpose: you can't embed an entire document as one vector and expect precise retrieval — a 10-page doc embedded as a single vector blurs every topic in it together. Chunking splits text into pieces small enough that each one has a *specific*, retrievable meaning. This is a real engineering decision, not boilerplate — chunk size and overlap directly control retrieval quality.

```python
def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap   # overlap prevents cutting a fact in half at a boundary
    return chunks

with open("docs/runbook.txt") as f:
    text = f.read()

chunks = chunk_text(text, chunk_size=30, overlap=8)  # small numbers here since our doc is tiny
for i, c in enumerate(chunks):
    print(f"--- chunk {i} ---\n{c}\n")
```
**What to notice:** each chunk is now small enough to represent one coherent idea (cert rotation, OR nginx restarts, OR autoscaling) instead of blending all three. The `overlap` exists so a sentence that straddles a chunk boundary still appears intact in at least one chunk — without it you risk splitting "restart nginx if health checks fail" across two chunks and losing the fact in both halves.

**Practical rule of thumb (not a hard law):** 200-500 tokens per chunk with 10-20% overlap is a common starting point for prose/runbooks. Code or structured data usually wants different splitting (function/class boundaries, not word counts) — a concern for later, not Week 2.

### Step 3 — Embed the chunks and store them (ingestion phase)

Purpose: this is the "chunk → embed → store" row of the diagram — done once per document set, not per query.

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_rag")
ef = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection("rag_docs", embedding_function=ef)

# Wipe and re-add for a clean run
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "runbook.txt", "chunk_index": i} for i in range(len(chunks))]
)
print(f"Stored {collection.count()} chunks")
```
**Why metadata matters here:** storing `source` and `chunk_index` alongside each vector means later you can show *which file and which part* an answer came from — this is what lets a real RAG app cite its sources instead of just hallucinating confidently. Treat this like adding structured labels to log lines so you can filter/trace later.

### Step 4 — Retrieval: turn a question into a search, and inspect what comes back *before* trusting it

Purpose: this is the most commonly skipped debugging step. Always look at what got retrieved before blaming the LLM for a bad answer — most "the AI is wrong" bugs are actually "the retrieval step returned the wrong chunk."

```python
def retrieve(question, n=2):
    results = collection.query(query_texts=[question], n_results=n)
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        print(f"[dist={dist:.3f}] ({meta['source']} #{meta['chunk_index']}) {doc[:80]}...")
    return results["documents"][0]

chunks_found = retrieve("what do I do if the cert doesn't renew automatically")
```
**What to check:** does the top result actually contain the cert-manager fallback command? If retrieval returns the nginx chunk instead, no amount of prompt engineering downstream will fix the answer — the bug is upstream. This is the single most important debugging habit in RAG work.

### Step 5 — Injection: build the augmented prompt (the actual "sidecar" step)

Purpose: this is where retrieved chunks get spliced into the prompt sent to the model. The structure of this prompt is itself a design decision, not boilerplate — a clear separation between "context" and "question" measurably improves answer grounding.

```python
def build_prompt(question, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

prompt = build_prompt("what do I do if the cert doesn't renew automatically", chunks_found)
print(prompt)
```
**Why the "ONLY the context" instruction matters:** without it, the model will happily blend its own generic training knowledge with your specific runbook, and you can no longer tell which parts are grounded in your real docs versus made up. This single line is your main lever against hallucination.

### Step 6 — Generation: send the augmented prompt to your local model

Purpose: close the loop — the model never sees your raw documents, only the already-augmented prompt. This is the "sidecar passes the enriched request through" step.

```python
import ollama

def generate(prompt, model="llama3.2"):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

answer = generate(prompt)
print(answer)
```

### Step 7 — Wire it into one function (the full pipeline, matching the diagram exactly)

```python
def rag_query(question, n=2, model="llama3.2", verbose=True):
    results = collection.query(query_texts=[question], n_results=n)
    context_chunks = results["documents"][0]

    if verbose:
        print("Retrieved:")
        for doc, dist in zip(context_chunks, results["distances"][0]):
            print(f"  [dist={dist:.3f}] {doc[:70]}...")

    prompt = build_prompt(question, context_chunks)
    return generate(prompt, model=model)

print(rag_query("how do I fix an nginx service that keeps failing health checks"))
```

### Step 8 — Prove RAG is doing something: same question, with and without it

Purpose: this is the practical proof step — run it once and you'll never forget why RAG exists.

```python
question = "what's the max replica count for autoscaling"

plain = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": question}])
print("WITHOUT RAG (model guesses from general training):")
print(plain["message"]["content"])

print("\nWITH RAG (grounded in your actual HPA config):")
print(rag_query(question, verbose=False))
```
**Expected result:** the plain model will give a generic, made-up-sounding answer about autoscaling in general. The RAG version will say "12" — because that specific number only exists in your runbook, not in the base model's training data. That gap *is* RAG's entire reason for existing.

### Step 9 — Tune it and watch quality change (this is where real engineering happens)

Purpose: RAG quality isn't fixed once you "have a pipeline" — chunk size, overlap, and `n` (how many chunks to retrieve) are knobs you tune against real questions, the same way you'd tune index parameters or cache TTLs.

```python
# Experiment: what happens with n=1 vs n=4 retrieved chunks?
for n in [1, 4]:
    print(f"\n--- n={n} ---")
    print(rag_query("what's the max replica count for autoscaling", n=n, verbose=True))
```
**What to watch for:** too few chunks (`n=1`) risks missing the answer if it's split across chunks. Too many (`n=4` on a 3-chunk corpus) starts injecting irrelevant context that can confuse the model or dilute the "ONLY use this context" instruction. On a real corpus, re-run Step 2 with different `chunk_size`/`overlap` values and re-observe retrieval quality in Step 4 — that's the actual tuning loop.

### Purpose recap — why each stage exists

| Stage | Purpose | Devops analogy |
|---|---|---|
| Chunking | Make retrieval precise instead of blurry | Splitting a monolith into addressable services |
| Embedding + store | Make chunks searchable by meaning | Building an index over your artifact store |
| Retrieval | Pull only what's relevant to this request | A cache lookup, but semantic instead of key-based |
| Injection | Attach the right context to the request | Sidecar mutating a request with config/secrets |
| Generation | Produce the final answer, grounded in real data | The main container, unaware of what the sidecar did |

### Use cases this unlocks

- **"Chat with our docs"** — internal wikis, runbooks, incident postmortems
- **Support/on-call assistant** — grounded in your actual playbooks, not generic advice
- **Code-aware assistant** — retrieve relevant functions/files before answering "how does X work in our repo"
- **Compliance/policy Q&A** — answers traceable back to a specific policy doc via metadata

---

## How the three fit together

- **Registry** (Day 1-4) → *which model* you're running
- **Vector DB** (Day 8) → *what to retrieve*
- **RAG** (Day 9) → *the glue that wires retrieval into generation at request time*

This is the architecture of the app built on Day 9 of the main curriculum — this file is the prerequisite reading for that day.
