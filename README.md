Constitution RAG with KEV as an LLM Judge

A Retrieval-Augmented Generation (RAG) system that uses KEV as a lightweight decision model / LLM judge between vector retrieval and final answer generation.

The project uses the Constitution of India as its knowledge base and demonstrates how KEV can evaluate the relevance of retrieved passages and rerank them before they are passed to the final generative LLM.

Why this project?

A conventional RAG pipeline usually looks like:

User Question
      |
      v
Vector Search
      |
      v
Top-K Documents
      |
      v
LLM
      |
      v
Answer

The problem is that semantic vector similarity does not necessarily mean that a retrieved passage is the best evidence for answering the question.

This project introduces KEV as a decision layer:

User Question
      |
      v
OpenAI Embedding
      |
      v
ChromaDB
      |
      | Top-K candidate passages
      v
+-----------------------+
|       KEV Judge       |
|                       |
| "How relevant is this |
|  passage to the       |
|  question?"           |
+-----------------------+
      |
      | KEV relevance scores
      v
Reranked Top-N Passages
      |
      v
OpenAI LLM
      |
      v
Grounded Answer

The main idea is:

ChromaDB retrieves candidates; KEV judges their usefulness; OpenAI generates the final answer.

Role of KEV

KEV is not being used as the final answer-generating LLM in this project.

Instead, KEV is used as a decision model / LLM judge.

Its job is to evaluate each retrieved passage against the user's question and produce a structured relevance score.

For example:

Question:
What does Article 14 of the Constitution provide?

Retrieved passage:
Article 14. Equality before law. —
The State shall not deny to any person equality before the law
or the equal protection of the laws within the territory of India.

KEV evaluates the passage using a structured score question:

{
  "type": "score",
  "instructions": "How relevant is this retrieved Constitution passage for answering the user's question?",
  "criteria": [
    "Not relevant",
    "Somewhat relevant",
    "Relevant",
    "Highly relevant"
  ]
}

The result includes a score, confidence, and probability distribution.

Example:

{
  "score": 2.85,
  "confidence": 0.91,
  "probabilities": {
    "0": 0.01,
    "1": 0.02,
    "2": 0.08,
    "3": 0.89
  }
}

The application uses this signal to rerank the retrieved passages.

KEV's Position in the Architecture

KEV sits between retrieval and generation.

                 RETRIEVAL                  JUDGING                 GENERATION

User Question
     |
     v
OpenAI Embedding
     |
     v
ChromaDB
     |
     | Top-K
     v
Candidate Passages
     |
     v
+----------------+
|      KEV       |
|     Judge      |
+----------------+
     |
     | Relevance scores
     v
Reranked Passages
     |
     v
OpenAI LLM
     |
     v
Final Answer

This separation gives each component a focused responsibility:

Component

Responsibility

OpenAI Embeddings

Convert questions and documents into vectors

ChromaDB

Retrieve semantically similar candidate passages

KEV

Judge passage relevance and rerank candidates

OpenAI LLM

Generate the final grounded response

FastAPI

Orchestrate the complete pipeline

Why use KEV instead of sending retrieved chunks directly to the LLM?

Suppose ChromaDB returns eight passages:

Candidate 1 → moderately relevant
Candidate 2 → unrelated
Candidate 3 → highly relevant
Candidate 4 → related
Candidate 5 → unrelated
Candidate 6 → highly relevant
Candidate 7 → weakly related
Candidate 8 → unrelated

A conventional RAG system may simply pass the top results to the generator based on vector similarity.

This project adds another decision step:

ChromaDB
    |
    | 8 candidates
    v
   KEV
    |
    | judge relevance
    v
4 best candidates
    |
    v
OpenAI

The objective is to reduce the amount of irrelevant or weak evidence reaching the generation model.

This makes KEV a reranking / evidence-selection layer rather than another chatbot.

End-to-End Flow

1. User asks a question

Example:

What does Article 14 of the Constitution of India provide?

2. Query embedding

The question is converted into an embedding using:

text-embedding-3-small

3. ChromaDB retrieval

ChromaDB searches the Constitution vector collection and returns the initial Top-K candidates.

For example:

Top 8 candidates

Each candidate contains:

{
  "chunk_id": "page-37-chunk-1",
  "text": "...",
  "metadata": {
    "page": 37,
    "chunk_index": 1,
    "source": "constitution_of_india.pdf"
  },
  "retrieval_distance": 0.79
}

4. KEV judging

Each candidate is independently sent to the local KEV server.

KEV receives the question together with the retrieved passage:

USER QUESTION:
What does Article 14 of the Constitution of India provide?

RETRIEVED CONSTITUTION PASSAGE:
...

KEV then evaluates the passage using a structured score question.

5. Reranking

The application sorts the candidates using the KEV relevance score.

Example:

Initial retrieval:

Passage A  → distance 0.72
Passage B  → distance 0.75
Passage C  → distance 0.77
Passage D  → distance 0.79
...


After KEV:

Passage C  → KEV 2.91
Passage A  → KEV 2.73
Passage F  → KEV 2.61
Passage D  → KEV 2.42

Only the selected Top-N passages are sent to the generation model.

6. Final answer generation

OpenAI receives:

User Question
+
KEV-selected constitutional passages

The generator is instructed to use only the supplied passages as evidence.

Project Architecture

constitution-rag-judge/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── ingest.py
│   ├── ingest_cli.py
│   ├── chroma_store.py
│   ├── retriever.py
│   ├── kev_judge.py
│   ├── reranker.py
│   └── generator.py
│
├── data/
│   └── constitution_of_india.pdf
│
├── chroma_db/
│
├── .env
├── .env.example
├── requirements.txt
└── README.md

kev_judge.py

Contains the integration with the local KEV server.

It sends requests to:

POST http://127.0.0.1:8009/v1/systemone

reranker.py

Coordinates KEV judging and sorts the retrieved candidates according to KEV's relevance score.

retriever.py

Handles:

Question
   ↓
OpenAI embedding
   ↓
ChromaDB
   ↓
Retrieved chunks

generator.py

Uses the KEV-selected passages as context for the final OpenAI generation step.

main.py

Exposes the FastAPI endpoints and orchestrates:

retrieve → judge → rerank → generate

Running KEV

KEV runs as a separate local service.

From the KEV repository:

uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-0.8b \
  --port 8009

The RAG project then communicates with KEV over HTTP.

KEV
localhost:8009

The Constitution RAG API runs independently:

FastAPI
localhost:8000

Therefore:

                 HTTP
FastAPI --------------------> KEV
:8000                         :8009

This separation is intentional.

The KEV repository does not need to be copied into this project.

Running the Project

1. Create the environment

Python 3.13 is recommended.

python3.13 -m venv .venv
source .venv/bin/activate

2. Install dependencies

python -m pip install -r requirements.txt

3. Configure environment variables

cp .env.example .env

Configure:

OPENAI_API_KEY=your_openai_api_key

OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

KEV_API_URL=http://127.0.0.1:8009
KEV_MODEL=kev-latest

CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=constitution_of_india

Never commit .env.

Add the Constitution

Place the Constitution PDF at:

data/constitution_of_india.pdf

Then run:

python -m app.ingest_cli

The ingestion process:

PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
OpenAI embeddings
 ↓
ChromaDB

Start FastAPI

python -m uvicorn app.main:app --reload --port 8000

API documentation:

http://127.0.0.1:8000/docs

API

Health

GET /health

Example:

curl http://127.0.0.1:8000/health

Response:

{
  "status": "ok",
  "kev_api_url": "http://127.0.0.1:8009",
  "chroma_collection": "constitution_of_india",
  "indexed_chunks": 500
}

Ask a Question

POST /ask

Example:

curl -s http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does Article 14 of the Constitution of India provide?",
    "retrieve_k": 8,
    "rerank_k": 4
  }'

The response contains:

{
  "question": "...",
  "answer": "...",
  "retrieved_chunks": [],
  "reranked_chunks": []
}

This makes it possible to inspect both stages:

retrieved_chunks
        ↓
     ChromaDB

reranked_chunks
        ↓
       KEV

Example KEV Decision

For a question about Article 14, ChromaDB may return passages concerning:

Article 14
Article 15
Article 19
Article 32
Article 312
...

KEV evaluates each passage independently.

The application then uses the KEV scores to determine which passages should be forwarded to the generation model.

The important distinction is:

Vector similarity ≠ final relevance

ChromaDB answers:

Which passages are semantically similar to my query?

KEV answers:

Given the question, how useful is this particular passage as evidence for answering it?

The combination is:

Semantic Retrieval
       +
Decision-based Relevance Judging
       =
Reranked RAG Context

Why this is interesting

This project demonstrates a pattern where a small decision model can be used as a specialized control/evaluation layer around a larger generative model.

Instead of using one LLM for everything:

Large LLM
 ├── retrieve
 ├── judge
 ├── reason
 └── generate

the responsibilities are separated:

Embedding Model → Retrieval

KEV → Decision / Relevance Judging

Generative LLM → Answer Generation

This provides a clear boundary between:

finding candidate evidence

deciding which evidence is useful

generating the final response

Current Limitations

This project is a prototype for experimenting with KEV-based RAG reranking.

Important limitations include:

KEV is a relevance signal, not a guaranteed correctness metric.

Retrieved passages are currently judged independently.

The current chunking strategy is based on PDF text chunks rather than fully structured constitutional Articles.

KEV inference adds latency because candidates are judged before generation.

The project should be evaluated on a fixed question/evidence dataset before making claims about improvements over baseline RAG.

The KEV project documentation notes limitations including calibration and sensitivity to option ordering.

Future Experiments

The project can be extended into an actual RAG evaluation framework.

1. Baseline vs KEV RAG

Compare:

Baseline RAG

Question
   ↓
ChromaDB
   ↓
OpenAI

against:

KEV RAG

Question
   ↓
ChromaDB
   ↓
KEV
   ↓
OpenAI

Measure:

retrieval quality

answer correctness

evidence relevance

latency

token usage

generation cost

2. Article-aware chunking

Instead of generic chunks:

page → chunk

create structured records:

Part
 └── Chapter
      └── Article
           └── Clause

This would allow the RAG system to preserve constitutional structure in metadata.

3. KEV judge calibration

Create a labeled evaluation dataset:

Question
Candidate Passage
Human Relevance Label
KEV Score

Then measure how well KEV's scores correlate with human relevance judgments.

4. Different judging criteria

Experiment with criteria such as:

0 = Irrelevant
1 = Related but insufficient
2 = Useful evidence
3 = Direct evidence

and compare them against broader criteria.

5. Multi-stage retrieval

A larger retrieval pool could be used:

ChromaDB Top-20
       ↓
KEV
       ↓
Top-5
       ↓
OpenAI

This gives KEV a larger candidate pool while limiting the amount of context sent to the final generator.

Key Takeaway

The core concept of this project is simple:

             RETRIEVE              JUDGE              GENERATE

Question ───────> ChromaDB ───────> KEV ───────> OpenAI
                    │                 │
                    │                 │
                 Find similar      Select useful
                  passages          evidence

ChromaDB finds the candidates.

KEV decides which candidates are relevant enough to keep.

OpenAI generates the final answer from the KEV-selected evidence.

That makes KEV the decision and reranking layer of the RAG pipeline, rather than another generative model.