# The Algorithmic Instructional Designer

An adaptive AI teacher built from small cooperating agents and a rule-based Orchestrator.
It plans a lesson, teaches one idea at a time, checks every answer, and **changes its teaching
method when the learner gets stuck** (hint, then simpler words, then going back, then a worked example).

Built with [Streamlit](https://streamlit.io) and the [Groq](https://groq.com) API
(model: `openai/gpt-oss-120b`).

## What it does

1. **Learner Profiler** collects level, goal and learning style, with an optional 3-question quick check.
2. **Architect** plans the lesson with Gagné's Nine Events of Instruction.
3. **Content Agent** writes one small concept at a time using Merrill's First Principles.
4. **Evaluator** checks written answers against a checklist. Multiple-choice answers are checked by plain code.
5. **Orchestrator** (simple if/else rules) decides the next move: continue, give a hint, change the style, go back, or show a worked example.
6. Progress is saved in SQLite so a learner can resume. You can also upload a PDF and the lesson is built only from it.
7. The sidebar shows cumulative input/output tokens and generated response characters for the lesson. After each answer, the feedback screen shows the usage for generating that question and evaluating the answer.

## How to use the app

1. Open the app.
2. In the **left sidebar**, paste your Groq API key and press Enter. Get a free key at <https://console.groq.com/keys>.
3. Type your name and a topic (or upload a PDF), then press **Start lesson**.

The key is typed by each visitor and kept only in that visitor's browser session. It is never written to the code, the database or GitHub.

## Files

```
.
├── app.py                  # the whole app (agents, database, interface)
├── requirements.txt        # Python packages
├── README.md
└── .streamlit/
    └── config.toml         # dark theme and colours (keep this folder name exactly)
```

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

## Deploy on Streamlit Community Cloud (free)

1. Create a new GitHub repository and upload all the files above, including the `.streamlit` folder.
   The folder name starts with a dot and must sit in the **root** of the repository, next to `app.py`.
   If it is missing, the app still opens, but the widgets may not match the dark design.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set **Main file path** to `app.py`.
4. Under **Advanced settings**, pick Python **3.12** (3.11 or 3.13 also work).
5. Click **Deploy**. When it finishes, you get a public link that opens straight into the app. There is no access code.

### Optional: use your own key for everyone

By default every visitor pastes their own key. If you want the app to work without that, add this under
**App settings > Secrets** on Streamlit Cloud:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Everyone using your link then spends **your** Groq quota. A key typed in the sidebar always wins over the secret.
Never put a real key in a file that you upload to GitHub.

## Settings you can change

Open `app.py` and edit the values near the top:

| Setting | Meaning | Default |
|---|---|---|
| `MODEL` | Groq model used by every agent | `openai/gpt-oss-120b` |
| `NUM_CONCEPTS` | Concepts planned per lesson (plus a final challenge) | `3` |
| `MAX_ATTEMPTS_PER_CONCEPT` | Safety limit for the teaching loop | `8` |
| `LESSON_LANGUAGE` | Language of lessons and feedback | `English` |
| `CHUNK_SIZE` | Characters per PDF chunk | `1500` |

## Good to know

- **Saved progress is temporary on Streamlit Cloud.** The SQLite file (`tutor.db`) is erased when the app restarts or redeploys.
- **Progress is saved under the name a learner types.** Anyone who types the same name on the same app can resume that lesson, so do not use this version for private data.
- Apps on the free tier go to sleep after a period of inactivity. Open the link and press the wake-up button.
- The app uses a dark theme only. `config.toml` also hides the viewer's theme switcher so nobody can switch it to light.
- If you see "Groq rejected this API key", paste the key again. It must start with `gsk_`.
- Scanned PDFs without selectable text cannot be read.

## Project summary

Agentic AI project, Version 1 (MVP). Learning-design ideas: Gagné's Nine Events of Instruction and Merrill's First Principles of Instruction.
