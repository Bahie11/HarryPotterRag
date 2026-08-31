# Harry Potter RAG Chatbot ⚡

A Retrieval-Augmented Generation (RAG) system that answers questions about the
Harry Potter books. It parses the full seven-book PDF, embeds it locally into a
vector database, and serves grounded answers — with source citations — through a
FastAPI backend and a chat UI.

Everything runs from a single all-in-one notebook
([`Harry_Potter_RAG_Notebook.ipynb`](Harry_Potter_RAG_Notebook.ipynb)), designed
to run end-to-end in Google Colab (no Docker, no separate server to start).

## How it works

```
harrypotter.pdf
   │  parse (PyMuPDF) → detect 7 books from the ToC
   ▼
clean & chunk  (fix drop-caps, de-hyphenate, keep chapter titles;
   │            sentence-aware chunks of ~800 chars, 120 overlap)
   ▼
embed locally  (BAAI/bge-small-en-v1.5, 384-dim, cosine)
   │
   ▼
Qdrant (embedded, on-disk)  ──►  FastAPI  /chat  ──►  Gradio / Streamlit UI
```

A user query flows through the API like this:

1. **Route** the query — a fast regex + LLM classifier labels it `greeting`,
   `harry_potter`, or `out_of_scope`, so small talk and off-topic questions skip
   retrieval.
2. **Retrieve** the top-K relevant chunks from Qdrant (query embedded with the
   BGE search prefix).
3. **Generate** an answer with the LLM, constrained to only use the retrieved
   passages, and return the **sources** (book + chapter + score) alongside it.

## Tech stack

| Layer          | Choice                                             |
| -------------- | -------------------------------------------------- |
| PDF parsing    | PyMuPDF (`fitz`)                                    |
| Embeddings     | `sentence-transformers` — `BAAI/bge-small-en-v1.5` |
| Vector store   | Qdrant (embedded mode, cosine distance)            |
| API            | FastAPI + Uvicorn                                  |
| LLM            | Any OpenAI-compatible endpoint (default: Groq `openai/gpt-oss-20b`) |
| UI             | Gradio (`share=True`) or Streamlit via Cloudflare tunnel |

Embeddings run locally, so the only credential required is an LLM API key
(a free key from https://console.groq.com works).

## Running it

1. Open [`Harry_Potter_RAG_Notebook.ipynb`](Harry_Potter_RAG_Notebook.ipynb) in
   Google Colab. *(Optional: `Runtime → Change runtime type → T4 GPU` makes
   embedding ~10× faster.)*
2. Run the cells top to bottom:
   - **Install dependencies**
   - **Upload the dataset** ([`harrypotter.pdf`](harrypotter.pdf)) and paste your
     LLM API key (read via `getpass`, never stored in the notebook).
   - **Parse → clean → chunk**, then **embed & index** into Qdrant.
   - **Launch the FastAPI server** in a background thread and query it over HTTP.
   - **Launch the Gradio / Streamlit UI** for an interactive chat.

The notebook also writes out `rag_api.py` — the standalone FastAPI server — so it
can be committed and reused outside Colab.

## API

`POST /chat`

```json
{ "query": "How did Harry get his Firebolt, and who sent it?", "top_k": 5 }
```

Response includes the `route`, the generated `answer`, and the `sources` used.
A `GET /health` endpoint reports readiness.

## Repository contents

- [`Harry_Potter_RAG_Notebook.ipynb`](Harry_Potter_RAG_Notebook.ipynb) — the
  complete pipeline: parse, embed, serve, and chat.
- [`harrypotter.pdf`](harrypotter.pdf) — the source dataset (all seven books).

## Notes

- Embedded Qdrant lives in `/content` on Colab and is wiped when the runtime
  resets. To persist it, mount Google Drive and point `QDRANT_PATH` at a Drive
  folder.
