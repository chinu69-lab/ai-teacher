# Algorithmic Instructional Designer

An adaptive AI tutor built with Streamlit, Python, and the Groq API. It plans a short lesson, teaches one concept at a time, checks answers, and changes its teaching method when the learner struggles.

## Learner experience

Learners enter a name and topic, optionally upload source notes, and choose their level, learning goal, and preferred learning style. The app can start with a three-question prior-knowledge check. During the lesson it presents explanations and examples, asks multiple-choice or written questions, evaluates answers, and adapts subsequent teaching. Progress is stored in SQLite so a lesson can be resumed.

The learning-style choice is visible during the lesson and guides the generated explanations and questions. Options include simple explanations, real-life examples, step-by-step practice, hands-on projects, interactive Q&A, quizzes and challenges, visual explanations, theory followed by practice, exam and interview preparation, and analogies and storytelling.

## How the AI agents work

- **Learner Profiler:** creates an optional quick check and uses the result to adjust the learner's starting level.
- **Architect:** divides the topic into concepts in a teaching order, using Gagné's Nine Events of Instruction.
- **Content Agent:** teaches one concept using Merrill's First Principles and shapes the explanation and question around the learner's selected style.
- **Evaluator:** checks written responses against a checklist; multiple-choice answers are scored in Python.
- **Orchestrator:** uses rule-based logic to choose the next step, such as continuing, giving a hint, retrying, revisiting an earlier concept, or showing a worked example.

The Groq model is configured by `MODEL` near the top of `app.py`. A learner supplies a Groq API key in the sidebar; the key is held in the Streamlit session and is not written to the SQLite database.

## Observability

Open **Observability** in the sidebar to inspect document extraction/chunking, model calls, judge assessments, and retrieval quality. The selected lesson summary displays the learner, topic, uploaded filename, extraction status, lesson stage, and profile settings. The selector recovers saved lesson details for the current learner when the live page state is empty. Call traces record a random run ID, agent name, model, success/error status, retry count, latency, and token counts. Telemetry tables do not store prompts, learner answers, topics, or retrieved document text.

New lesson segments are scored by an LLM-as-Judge using overall, accuracy, relevance, clarity, learner-fit, and safety scores from 1 to 5. The default model ID is `qwen/qwen3.8-27b`; confirm it is available to your Groq account or override it with `GROQ_JUDGE_MODEL`. The judge adds one Groq call per generated segment and uses the learner's Groq key. A rejected model ID or judge error is recorded but does not block lesson generation. Scores are advisory, not ground truth.

The Observability dashboard features a conversation selector allowing inspection of the active lesson or previous learner conversations. It is structured into four tabs: RAG Pipeline Quality, AI Guardrails & Safety, Agent Call Telemetry, and LLM-as-a-Judge Quality. RAG evaluation is 100% automated for the main application: whenever notes are used in a lesson, retrieval precision@k and recall@k are automatically calculated for each concept and displayed with BM25, embedding, and RRF fusion scores both on the active learning screen and in the Observability tab. No manual measuring is required.

Automated evaluation assesses whether retrieved chunks contain the key facts required to teach each concept. Evaluate multiple representative documents before altering chunk size, overlap, embedding model, or retrieval parameters.

## AI guardrails

Every model call receives shared safety and trust-boundary instructions. When NeMo Guardrails is installed and enabled, its input and output self-check rails check model requests and responses, except the topic-review request: this call has its own topic-safety policy and makes the explicit topic decision, and its response is validated against the `TopicReview` schema. NeMo connects to Groq's OpenAI-compatible endpoint using the session's Groq key; the Observability page shows the sanitized config and pass/block/error trace. A NeMo service error is recorded and the request continues under the app's prompt-level safeguards; an explicit NeMo safety block on other input or output still stops the request. Learner text and uploaded notes are treated as untrusted content, not instructions. Topic review rejects unclear requests and requests for actionable help causing harm, self-harm, crimes, exploitation, or evading safeguards, while allowing safe high-level, historical, preventive, and defensive education. Model responses are also parsed and checked against Pydantic schemas and question-format rules before use.

Install `nemoguardrails` from `requirements.txt` in the Python environment that runs Streamlit. If the package is unavailable, the app reports that state in Observability and continues with the existing prompt-level safeguards. These checks reduce risk but are not a guarantee that every model response is safe or correct. Review generated lessons, and do not enter credentials or sensitive personal information in topics, notes, or answers.

## Uploaded documents

The uploader accepts PDF, PowerPoint (`.pptx`), Word (`.docx` and legacy `.doc`), OpenDocument Text (`.odt`), plain text (`.txt`), and RTF (`.rtf`). The app extracts readable text from these files and uses it as lesson source material. Legacy `.doc` files also require the `antiword` utility installed on the computer running the app.

Image files such as PNG, JPG, JPEG, and WEBP are **not accepted**. The PDF reader extracts text blocks and skips image blocks; the PowerPoint reader extracts text and tables but does not read text embedded in images. Therefore scanned PDFs and image-only slides are not usable as notes. OCR and direct image understanding are not currently implemented.

## Why hybrid retrieval is used

When an uploaded source is available, the app retrieves relevant text chunks for each concept. It combines two complementary search methods:

- **BM25 keyword retrieval** is good at exact words, names, and technical terms.
- **Embedding retrieval** uses the `BAAI/bge-small-en-v1.5` model to find passages that are semantically related even when they use different wording.
- **Reciprocal Rank Fusion (RRF)** combines the two ranked lists. This avoids comparing raw keyword and vector scores, which use different scales, and helps preserve results found by either method. The app then supplies the top two retrieved chunks to the content writer.

Using both methods helps when a learner's question uses different phrasing from the uploaded notes while still retaining strong exact-term matches. Retrieval uses the `rank-bm25` package and local embeddings from `fastembed`.

## Why recursive chunking is used

Long documents are split into manageable pieces before search. The splitter tries paragraph breaks first, then sentence boundaries, then word boundaries, and uses a character limit only when a longer unit cannot be split at a preferred boundary. The current maximum chunk size is 1,500 characters, with 150 characters of overlap between adjacent chunks.

This recursive strategy keeps related text together where possible, while the overlap carries a little context across chunk boundaries. That reduces the chance that a definition and its nearby explanation are separated in a way that makes retrieval less useful.

## Project files and settings

- `app.py` contains the Streamlit interface, agents, document readers, retrieval functions, SQLite persistence, and orchestration rules.
- `requirements.txt` lists Python dependencies.
- `config.toml` and `.streamlit/config.toml` contain Streamlit theme configuration; the app uses `.streamlit/config.toml` when deployed from the repository root.
- `README.md` contains setup and deployment instructions.
- `tutor.db` is created at runtime and stores saved learner state and answer history.

Important settings near the top of `app.py` include the model, number of lesson concepts, language, chunk size, chunk overlap, embedding model, and database path. The default database is local SQLite. On Streamlit Community Cloud, local files can be temporary and may be lost on restart or redeployment.

## Privacy and deployment notes

The app has no authentication. The Profiles & activity view can show saved learner names, topics, questions, answers, and results to visitors of the deployed app. Do not use it with private or sensitive learner data. Anyone using the same learner name may resume that saved profile.

Install the packages with `pip install -r requirements.txt`, then start the app with `streamlit run app.py`. The first source-based lesson may download the embedding model; later use can reuse the cached model. For deployment, follow the Streamlit Community Cloud steps in `README.md` and store any owner-provided Groq API key in Streamlit secrets, never in source code.