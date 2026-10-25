# The Algorithmic Instructional Designer

An adaptive AI teacher built from small cooperating agents and a rule-based Orchestrator. It plans a lesson, teaches one idea at a time, checks every answer, and changes its teaching method when the learner gets stuck.

Built with [Streamlit](https://streamlit.io) and the [Groq](https://groq.com) API (model: `openai/gpt-oss-120b`).

## What it does

1. **Learner Profiler** collects level, goal and learning style, with an optional 3-question quick check.
2. **Architect** plans the lesson with Gagné's Nine Events of Instruction.
3. **Content Agent** writes one small concept at a time using Merrill's First Principles.
4. **Evaluator** checks written answers against a checklist. Multiple-choice answers are checked by plain code.
5. **Orchestrator** (simple if/else rules) decides the next move: continue, give a hint, change the style, go back, or show a worked example.
6. Progress is saved in SQLite so a learner can resume. Supported notes include PDF, PowerPoint (`.pptx`), Word (`.docx` and `.doc`), OpenDocument Text (`.odt`), plain text (`.txt`), and RTF (`.rtf`).
7. The sidebar shows cumulative input/output tokens and generated response characters. Feedback shows usage for generating a question and evaluating its answer.
8. The app checks whether a topic is clear and compatible with uploaded notes before planning a lesson.
9. Uploaded notes use hybrid retrieval: BM25 finds exact keyword matches, BGE-small embeddings find semantically related passages, and Reciprocal Rank Fusion combines the rankings.
10. **Profiles & activity** summarizes saved profiles and question history.
11. The top-right **Project guide** opens project documentation and offers a Markdown download.
12. **Observability** shows agent traces, LLM-as-Judge quality scores, extracted chunks, BM25/vector/RRF rankings, and human-labeled precision@k/recall@k evaluations.
13. NeMo Guardrails can run input and output safety checks through Groq; its decisions are visible in Observability.

## How to use the app

1. Open the app.
2. In the left sidebar, paste your Groq API key and press Enter. Get a key at <https://console.groq.com/keys>.
3. Type your name and a topic (or upload supported notes), then press **Start lesson**.

The key is typed by each visitor and held in that visitor's Streamlit session. It is not written to the code, database, or GitHub.

## Files

```
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- PROJECT_GUIDE.md
|-- .gitignore
`-- .streamlit/
    `-- config.toml
```

## Run on Windows

Use this workspace's own environment so dependencies do not get installed into another project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Reading older Word `.doc` files also requires the `antiword` system utility on `PATH`. `.docx` files do not need it.

The first lesson using uploaded notes downloads the `BAAI/bge-small-en-v1.5` embedding model. The model is cached for later retrievals while the app process is running.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload the project files, including the `.streamlit` folder.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set the main file path to `app.py`.
4. Under **Advanced settings**, choose Python 3.12 (3.11 or 3.13 also work).
5. Click **Deploy**. When deployment finishes, you get a public link.

### Optional: use your own key for everyone

By default each visitor pastes their own key. To use an owner-provided key, add this under Streamlit Cloud app secrets:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Everyone using the app then spends the owner's Groq quota. A key typed in the sidebar takes precedence. Never put a real key in a file uploaded to GitHub.

## Settings

Open `app.py` and edit settings near the top:

| Setting | Meaning | Default |
|---|---|---|
| `MODEL` | Groq model used by lesson agents and NeMo rails | `openai/gpt-oss-120b` |
| `GROQ_JUDGE_MODEL` | Optional environment override for the judge | `qwen/qwen3.8-27b` |
| `NUM_CONCEPTS` | Concepts planned per lesson, plus a final challenge | `3` |
| `MAX_ATTEMPTS_PER_CONCEPT` | Safety limit for the teaching loop | `8` |
| `LESSON_LANGUAGE` | Language of lessons and feedback | `English` |
| `CHUNK_SIZE` | Maximum characters per document chunk | `1500` |
| `CHUNK_OVERLAP` | Context characters carried between chunks | `150` |
| `EMBEDDING_MODEL` | Local semantic retrieval model | `BAAI/bge-small-en-v1.5` |

## Good to know

- **Profiles & activity is public.** Anyone who visits the app can see learner names, profile settings, topics, questions, answers, and results. Do not use it with private or sensitive learner information.
- A profile is identified by learner name. Reusing a name resumes or overwrites that profile.
- Saved SQLite progress is temporary on Streamlit Community Cloud and may be erased when the app restarts or redeploys.
- The app has no authentication. Anyone using the same learner name on the same app can resume that lesson.
- Scanned PDFs and image-only slides cannot be read; OCR and direct image input are not implemented.
- Documents are split recursively at paragraph, sentence, and word boundaries, with overlap between adjacent chunks.
- Hybrid retrieval combines BM25 exact-term search and embedding semantic search using RRF.
- NeMo Guardrails checks use the configured Groq model and can add up to two model calls per agent call, increasing latency and token usage. Judge scores also add a Groq request per generated lesson segment.
- LLM-as-Judge scores are advisory, not ground truth. The exact Groq judge model ID must be enabled for your account.

## Project guide

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for architecture, RAG details, guardrails, observability, privacy notes, and documented limitations.

Agentic AI project, Version 1 (MVP). Learning-design ideas: Gagné's Nine Events and Merrill's First Principles.
