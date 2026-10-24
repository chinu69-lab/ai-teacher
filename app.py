# =====================================================================
#  THE ALGORITHMIC INSTRUCTIONAL DESIGNER  -  app.py
#  An adaptive AI teacher made of small agents + a rule-based Orchestrator.
#  Run locally:  streamlit run app.py      Deploy: Streamlit Community Cloud (see README.md)
# =====================================================================
import os, re, json, time, copy, random, sqlite3, html
from contextlib import contextmanager
from typing import List, Optional, Literal

import streamlit as st
from groq import Groq, AuthenticationError
from pydantic import BaseModel, Field, ValidationError

# ---------------- SETTINGS (edit these values to change the behaviour) ----------------
MODEL = "openai/gpt-oss-120b"        # Groq model used by every agent
NUM_CONCEPTS = 3                      # how many concepts the Architect plans (+1 final challenge)
MAX_ATTEMPTS_PER_CONCEPT = 8          # safety limit: the teaching loop can never run forever
LESSON_LANGUAGE = "English"           # language of lessons, questions and feedback
CHUNK_SIZE = 1500                     # characters per PDF chunk
DB_PATH = os.environ.get("DB_PATH", "tutor.db")            # SQLite file (on Streamlit Cloud it resets when the app restarts)

# ---------------- ONE PLACE THAT TALKS TO THE LLM ----------------
# Every agent calls ask_json(). To change the model/provider later, only this part changes.
def current_key():
    # 1) the key typed in the sidebar (private to this visitor's browser session)
    # 2) optional fallback for the app owner: GROQ_API_KEY environment variable or Streamlit secret
    key = (st.session_state.get("groq_key") or "").strip()
    if not key:
        key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            key = ""
    return key


def get_client():
    # One Groq client per visitor session. It is rebuilt whenever the key changes.
    key = current_key()
    if not key:
        raise RuntimeError("Add your Groq API key in the sidebar first.")
    cached = st.session_state.get("_groq")
    if cached is None or cached[0] != key:
        cached = (key, Groq(api_key=key))
        st.session_state["_groq"] = cached
    return cached[1]


def parse_json_text(text):
    # Models sometimes wrap JSON in ```json fences. Remove them and read the first {...} block.
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def ask_json(system_prompt, user_prompt, temperature=0.3, retries=3):
    # Ask the model and return a Python dict. Retries on rate limits (429) and bad JSON.
    client = get_client()                            # raises a clear message when the key is missing
    use_json_mode = True
    last_error = None
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt + "\n\nReply with valid JSON only."},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            reply = client.chat.completions.create(**kwargs)
            return parse_json_text(reply.choices[0].message.content)
        except AuthenticationError:                  # wrong key: retrying cannot help
            raise RuntimeError("Groq rejected this API key. Check the key in the sidebar and try again.")
        except Exception as e:                       # rate limit, network, or bad JSON
            last_error = e
            if "response_format" in str(e).lower():  # model does not support JSON mode -> retry without it
                use_json_mode = False
            time.sleep(2 * (attempt + 1))            # wait a little longer each time
    raise RuntimeError(f"The AI model failed after {retries} tries: {last_error}")

# ---------------- DATA SHAPES (Pydantic) + QUESTION CHECKER ----------------
# Pydantic makes sure the AI's JSON has the right fields before we use it.
class Concept(BaseModel):
    title: str
    hook: str = ""       # Gagne step 1: gain attention
    goal: str = ""       # Gagne step 2: tell the learning goal
    recall: str = ""     # Gagne step 3: recall prior knowledge

class LessonPlan(BaseModel):
    topic: str = ""
    concepts: List[Concept]

class Question(BaseModel):
    kind: Literal["mcq", "short"]
    text: str
    options: List[str] = Field(default_factory=list)      # only for mcq
    correct_index: Optional[int] = None                   # only for mcq
    sample_answer: str = ""                               # only for short
    checklist: List[str] = Field(default_factory=list)    # only for short
    explanation: str = ""

class ConceptContent(BaseModel):
    explanation: str
    example: str = ""
    hint: str = ""
    source_quote: str = ""      # a sentence copied from the PDF (used to verify PDF lessons)
    question: Question

class EvalResult(BaseModel):
    result: Literal["correct", "partial", "wrong", "unclear"]
    right: str = ""
    missing: str = ""
    misconception: str = ""
    feedback: str
    follow_up: str = ""


def check_question(q, kind):
    # NORMAL CODE (no AI): returns a list of problems. Empty list = the question is fine.
    problems = []
    if q.kind != kind:
        problems.append(f"question kind must be '{kind}'")
    if not q.text.strip():
        problems.append("question text is empty")
    if kind == "mcq":
        opts = [o.strip() for o in q.options]
        if len(opts) != 4:
            problems.append("an MCQ needs exactly 4 options")
        if "" in opts or len(set(o.lower() for o in opts)) != len(opts):
            problems.append("options must be different and not empty")
        if q.correct_index is None or not (0 <= q.correct_index < len(opts)):
            problems.append("correct_index must point to one option")
    else:
        if not q.sample_answer.strip():
            problems.append("short question needs a sample answer")
        if not q.checklist:
            problems.append("short question needs a checklist")
    return problems


def shuffle_options(q):
    # LLMs like to put the right answer in the same place, so we shuffle the 4 options.
    right = q.options[q.correct_index]
    random.shuffle(q.options)
    q.correct_index = q.options.index(right)
    return q

# ---------------- SQLITE DATABASE (progress is saved after every step) ----------------
@contextmanager
def db():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    # sessions = the full lesson state of each learner (so they can resume)
    # answers  = one row per answered question (useful for analytics later)
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS sessions "
                    "(learner TEXT PRIMARY KEY, state_json TEXT, updated REAL)")
        con.execute("CREATE TABLE IF NOT EXISTS answers "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, learner TEXT, topic TEXT, concept TEXT, "
                    "kind TEXT, question TEXT, answer TEXT, result TEXT, created REAL)")


def save_progress(S):
    if not S.get("learner"):
        return
    with db() as con:
        con.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
                    (S["learner"], json.dumps(S), time.time()))


def load_progress(learner):
    with db() as con:
        row = con.execute("SELECT state_json FROM sessions WHERE learner = ?", (learner,)).fetchone()
    return json.loads(row[0]) if row else None


def log_answer(S, concept, q, answer, result):
    with db() as con:
        con.execute("INSERT INTO answers (learner, topic, concept, kind, question, answer, result, created) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (S["learner"], S["topic"], concept, q["kind"], q["text"], str(answer)[:500], result, time.time()))


def delete_learner(learner):
    # The learner can delete all of their data
    with db() as con:
        con.execute("DELETE FROM sessions WHERE learner = ?", (learner,))
        con.execute("DELETE FROM answers WHERE learner = ?", (learner,))

# ---------------- PDF READING AND SIMPLE RETRIEVAL ----------------
def read_pdf(file_bytes, filename="notes.pdf"):
    # Read text with PyMuPDF, keep paragraphs, collect short heading-like lines, make chunks.
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    paragraphs, headings = [], []
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] != 0:                       # 0 = text block (skip images)
                continue
            text = " ".join(block[4].split())
            if not text:
                continue
            paragraphs.append(text)
            looks_like_heading = (3 < len(text) < 80 and not text.endswith((".", ",", ";"))
                                  and any(ch.isalpha() for ch in text))
            if looks_like_heading and len(headings) < 40:
                headings.append(text)
    if sum(len(p) for p in paragraphs) < 200:
        raise ValueError("no readable text found (is it a scanned PDF?)")
    chunks, current = [], ""
    for p in paragraphs:                            # join paragraphs into chunks of ~CHUNK_SIZE characters
        if current and len(current) + len(p) > CHUNK_SIZE:
            chunks.append(current.strip())
            current = ""
        current += p + "\n"
    if current.strip():
        chunks.append(current.strip())
    return {"name": filename, "chunks": chunks, "headings": headings}


def material_overview(material, max_chars=6000):
    # A short overview of the whole PDF for the Architect (headings + a few chunks)
    if not material:
        return ""
    text = "Headings found: " + "; ".join(material["headings"][:25]) + "\n\n"
    chunks = material["chunks"]
    step = max(1, len(chunks) // 6)
    for c in chunks[::step]:
        if len(text) + 800 > max_chars:
            break
        text += c[:800] + "\n---\n"
    return text


def pick_chunks(material, query, k=2):
    # Very simple retrieval: choose the chunks that share the most words with the concept.
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())]
    scored = []
    for i, chunk in enumerate(material["chunks"]):
        low = chunk.lower()
        scored.append((sum(low.count(w) for w in words), -i, chunk))
    scored.sort(reverse=True)
    return "\n---\n".join(c for _, _, c in scored[:k])


def quote_in_material(quote, material):
    # Hallucination check (normal code): is the quoted sentence really inside the PDF?
    norm = lambda t: " ".join(t.lower().split())
    quote = norm(quote)
    if len(quote) < 15 or not material:
        return False
    return quote in norm(" ".join(material["chunks"]))

# ---------------- AGENT 1: LEARNER PROFILER ----------------
LEVELS = ["beginner", "intermediate", "advanced"]
GOALS = ["exam", "interview", "project", "just interest"]
STYLE_PREFS = ["simple explanations", "real-life examples", "step-by-step practice"]

def default_profile():
    # Used when the learner skips the questions
    return {"level": "beginner", "goal": "just interest", "style": "real-life examples"}


def profile_text(p):
    return f"level = {p.get('level')}, goal = {p.get('goal')}, likes = {p.get('style')}"


def make_quiz(S):
    # The Profiler writes a 3-question quick check, because people do not always judge their level correctly.
    system = ("You are the LEARNER PROFILER agent of an AI teacher. Write 3 very short multiple-choice questions "
              f"in {LESSON_LANGUAGE} that check what the learner already knows about the topic (easy to medium). "
              'Return JSON: {"questions":[{"kind":"mcq","text":"...","options":["...","...","...","..."],'
              '"correct_index":0,"explanation":"..."}]}  Exactly 4 options, only one correct.')
    user = f"Topic: {S['topic']}\nClaimed level: {S['profile'].get('level')}"
    if S["material"]:
        user += "\nUse ONLY this source material:\n" + material_overview(S["material"], 3000)
    data = ask_json(system, user)
    good = []
    for raw in data.get("questions", []):
        try:
            q = Question.model_validate({**raw, "kind": "mcq"})
            if not check_question(q, "mcq"):
                good.append(shuffle_options(q).model_dump())
        except Exception:
            pass                                    # skip a broken question, keep the good ones
    return good[:3]


def apply_quiz(S, score):
    # If the learner claims a high level but fails the quick check, start from the basics.
    n = len(S["quiz"])
    S["profile"]["quiz_score"] = f"{score}/{n}"
    if score * 3 <= n and S["profile"]["level"] != "beginner":
        S["profile"]["level_note"] = f"was '{S['profile']['level']}', lowered after the quick check"
        S["profile"]["level"] = "beginner"

# ---------------- AGENT 2: ARCHITECT (lesson plan with Gagne's Nine Events) ----------------
def make_lesson_plan(S):
    system = ("You are the ARCHITECT agent of an AI teacher, an instructional designer. Plan a lesson using "
              "Gagne's Nine Events of Instruction. Break the topic into small concepts in teaching order "
              "(each idea builds on the one before). Beginners get simple definitions first; advanced learners "
              f"skip the basics. Write in {LESSON_LANGUAGE}. For every concept give: "
              "title, hook (a short question or real-life situation that gets attention), "
              "goal (what the learner will be able to do), recall (one line linking to something they probably already know). "
              'Return JSON: {"topic":"...","concepts":[{"title":"...","hook":"...","goal":"...","recall":"..."}]}')
    user = (f"Topic: {S['topic']}\nLearner: {profile_text(S['profile'])}\n"
            f"Number of concepts: exactly {NUM_CONCEPTS}")
    if S["material"]:
        user += ("\n\nSOURCE material (build the concepts ONLY from this, do not invent facts):\n"
                 + material_overview(S["material"]))
    for attempt in range(3):
        try:
            plan = LessonPlan.model_validate(ask_json(system, user))
            if plan.concepts:
                return [c.model_dump() for c in plan.concepts][:NUM_CONCEPTS]
        except ValidationError:
            continue
    raise RuntimeError("The Architect could not make a valid lesson plan. Please try again.")

# ---------------- AGENT 3: CONTENT AGENT (Merrill's First Principles) ----------------
# The Orchestrator picks a METHOD. The Content Agent writes the lesson part in that way.
METHODS = {
    "first_time":   "Teach this concept for the first time. Start from a small real-life problem, then explain simply.",
    "after_review": "The learner just reviewed an earlier idea. Teach this concept again, briefly, building on that idea.",
    "hint_retry":   "The learner answered wrong once. Give a SHORT re-explanation of the key point, a clear hint, and a NEW similar question (different wording).",
    "fill_gap":     "The learner was partly right. Explain ONLY the missing part clearly, then ask a NEW question about it.",
    "follow_up":    "The learner's answer was unclear. Do not re-teach. Keep the explanation to one or two lines and ask a NEW simple follow-up question to see what they really understand.",
    "change_style": "The learner is still struggling. Use VERY simple words and this style: {style}. Do not repeat the earlier explanation.",
    "go_back":      "The learner keeps failing the next idea, so we go back to this EARLIER idea. Give a quick recap and a simple question to check it.",
    "worked_example": "The learner failed several times. STOP testing. Teach step by step with ONE fully worked example (show every step), then ask an EASY question about it.",
    "final":        "This is the FINAL CHALLENGE (test and real-life use). Put a short recap in 'explanation' and a NEW real-life scenario in 'example'. The question must ask the learner to apply the ideas to that new situation.",
}
STYLE_IDEAS = ["a real-life analogy", "a brand-new example from daily life", "tiny step-by-step steps"]

QUESTION_FORMAT = {
    "mcq": ('"question" = {"kind":"mcq","text":"...","options":["...","...","...","..."],"correct_index":0,'
            '"explanation":"why the correct option is right"}. Exactly 4 options (no A/B/C letters), only one correct.'),
    "short": ('"question" = {"kind":"short","text":"...","sample_answer":"...","checklist":["point 1","point 2"],'
              '"explanation":"..."}. The checklist has 2 to 4 key points a good answer must contain.'),
}


def write_concept(S, concept, method, kind, extra="", style=""):
    p = S["profile"]
    material = S["material"]
    source = ""
    if material:
        source = pick_chunks(material, concept["title"] + " " + concept.get("goal", ""))
    system = (f"You are the CONTENT agent of an AI teacher. Write ONE small part of a lesson in {LESSON_LANGUAGE} "
              "using Merrill's First Principles: start from a real problem, connect to what the learner knows, "
              "show how to solve it, let the learner try. Keep it short and simple: the explanation is under 120 words "
              "and shows only one idea. Return JSON with the keys: explanation, example, hint, source_quote, question. "
              + QUESTION_FORMAT[kind] +
              " If SOURCE text is given, use ONLY facts from it and copy ONE supporting sentence word-for-word into "
              "source_quote. If there is no SOURCE, source_quote must be an empty string.")
    user = (f"Concept: {concept['title']}\nGoal: {concept.get('goal', '')}\n"
            f"Learner: {profile_text(p)}\n"
            f"Teaching method: {METHODS[method].replace('{style}', style)}\n")
    if method == "final":
        user += "Concepts taught in this lesson: " + "; ".join(c["title"] for c in S["plan"]) + "\n"
    if extra:
        user += f"What we know about the learner's problem: {extra}\n"
    earlier = S["asked"].get(concept["title"], [])
    if earlier:
        user += "Questions already asked (do NOT repeat them): " + " | ".join(earlier) + "\n"
    if source:
        user += "\nSOURCE:\n" + source
    last_error = None
    for attempt in range(3):
        try:
            content = ConceptContent.model_validate(ask_json(system, user))
            problems = check_question(content.question, kind)       # normal-code check
            if problems:
                raise ValueError("; ".join(problems))
            if kind == "mcq":
                shuffle_options(content.question)
            result = content.model_dump()
            if material:                                             # verify the lesson against the PDF
                result["source_status"] = "verified" if quote_in_material(content.source_quote, material) else "unverified"
            else:
                result["source_status"] = "none"
            return result
        except (ValidationError, ValueError) as e:
            last_error = e
    raise RuntimeError(f"The Content agent could not write a valid lesson part: {last_error}")

# ---------------- AGENT 4: EVALUATOR ----------------
def check_mcq(q, chosen_index):
    # NORMAL CODE (no AI): compare the pick with the saved answer. Fast and always reliable.
    ok = (chosen_index == q["correct_index"])
    return {"result": "correct" if ok else "wrong", "right": "", "missing": "", "misconception": "",
            "feedback": q["explanation"], "follow_up": "", "skipped": False}


def evaluate_answer(S, q, answer):
    # The Evaluator (AI) compares a written answer with the checklist - exact words are not needed.
    system = ("You are the EVALUATOR agent of an AI teacher. Compare the learner's answer with the checklist; "
              "the learner does not need the exact same words. Decide the result: "
              "'correct' (covers all key points), 'partial' (some key points but something important is missing), "
              "'wrong' (incorrect or off-topic), 'unclear' (too short or vague to judge). "
              "Be fair: ignore spelling and typing mistakes, and remember one weak answer does not prove the learner "
              f"did not understand. Write the feedback in {LESSON_LANGUAGE}, friendly, 2-3 simple sentences: what is right, "
              "what is missing, and the likely misunderstanding. If the result is 'unclear', also write one short follow_up question. "
              'Return JSON: {"result":"...","right":"...","missing":"...","misconception":"...","feedback":"...","follow_up":"..."}')
    user = (f"Question: {q['text']}\nSample answer: {q['sample_answer']}\n"
            "Checklist:\n- " + "\n- ".join(q["checklist"]) + f"\n\nLearner's answer: {answer}")
    for attempt in range(2):
        try:
            ev = EvalResult.model_validate(ask_json(system, user, temperature=0.1)).model_dump()
            ev["skipped"] = False
            return ev
        except ValidationError:
            continue
    return {"result": "unclear", "right": "", "missing": "", "misconception": "", "skipped": False,
            "feedback": "I had trouble judging that answer. Could you explain it again in a different way?",
            "follow_up": ""}

# ---------------- SHARED LESSON STATE ----------------
# All agents share ONE dictionary (the "state"). It is saved to SQLite after every step.
def new_state():
    return {
        "learner": "", "stage": "start",          # stage: start -> profile -> quiz -> learning -> done
        "topic": "", "material": None,
        "profile": {}, "quiz": [],
        "plan": [], "current": 0,                 # index of the current concept (len(plan) = final challenge)
        "content": None, "phase": "question",     # phase: question -> feedback
        "qcount": 0,
        "wrong_streak": 0, "attempts": 0, "rescue_used": False, "return_to": None,
        "methods_tried": [], "needs_revision": [], "completed": [],
        "asked": {}, "history": [], "last_eval": None, "action": None, "log": [],
    }


def is_final(S):
    return S["current"] >= len(S["plan"])


def current_concept(S):
    if is_final(S):
        return {"title": "Final challenge: " + S["topic"], "hook": "", "recall": "",
                "goal": "Use all the ideas together in a new real-life situation"}
    return S["plan"][S["current"]]


def reset_counters(S):
    S["wrong_streak"], S["attempts"], S["rescue_used"] = 0, 0, False


def mark_revision(S, title):
    # Save "needs revision" together with the teaching methods that were already tried
    tried = sorted({m["method"] for m in S["methods_tried"] if m["concept"] == title})
    for item in S["needs_revision"]:
        if item["concept"] == title:
            item["methods_tried"] = tried
            return
    S["needs_revision"].append({"concept": title, "methods_tried": tried})


def commit(S, T):
    # Copy the finished working copy T into the real state S and save it.
    S.clear()
    S.update(T)
    save_progress(S)


# ---------------- THE ORCHESTRATOR: plain if/else rules, NO AI ----------------
def decide_next(result, S):
    # Returns the next action. Rules follow the table in the project document.
    if is_final(S):
        return "finish"
    if S["attempts"] >= MAX_ATTEMPTS_PER_CONCEPT:
        return "skip_revision"
    if result == "correct":
        return "next_concept"
    if result == "unclear":
        return "follow_up"
    if result == "partial":
        return "fill_gap"
    streak = S["wrong_streak"]                      # result == "wrong" (or "I don't know")
    if streak == 1:
        return "hint_retry"
    if streak == 2:
        return "change_style"
    if streak == 3 and S["current"] > 0 and S["return_to"] is None:
        return "go_back"
    return "skip_revision" if S["rescue_used"] else "worked_example"


ACTION_TEXT = {
    "next_concept":  "Great, you understood this. Moving on to the next idea.",
    "follow_up":     "Your answer was not clear enough for me to judge, so I will ask a short follow-up question.",
    "fill_gap":      "I will explain the missing part and then ask you a new question.",
    "hint_retry":    "I will give you a hint and a similar question.",
    "change_style":  "I will teach this again in a different, simpler way and then test you again.",
    "go_back":       "I will go back to an earlier idea and check that first.",
    "worked_example": "I will stop testing now and teach step by step with a fully worked example, then ask an easy question. This topic is saved as 'needs revision'.",
    "skip_revision": "Let's move on for now. This topic is saved as 'needs revision' so we can review it next time. If it stays hard, please ask a teacher.",
    "finish":        "That was the last step. Let's look at your summary.",
}
METHOD_LABEL = {
    "hint_retry": "a hint and a similar question", "fill_gap": "explaining the missing part",
    "follow_up": "a follow-up question", "change_style": "simpler words and a different style",
    "go_back": "going back to an earlier idea", "worked_example": "a fully worked example",
    "after_review": "teaching again after the review",
}


def pick_kind(S, method):
    # Which question type to ask: MCQ or written answer
    if is_final(S) or method in ("fill_gap", "follow_up"):
        return "short"
    if method in ("first_time", "after_review"):
        return "short" if S["current"] % 2 == 1 else "mcq"      # mix both types
    return "mcq"


def pick_style(S):
    # Rotate the teaching style so we never repeat the same explanation
    title = current_concept(S)["title"]
    n = sum(1 for m in S["methods_tried"] if m["concept"] == title and m["method"] == "change_style")
    return STYLE_IDEAS[n % len(STYLE_IDEAS)]


def generate_content(S, method, extra="", style=""):
    # Ask the Content Agent for the next lesson part and put it into the state
    if is_final(S):
        method = "final"
    concept = current_concept(S)
    kind = pick_kind(S, method)
    content = write_concept(S, concept, method, kind, extra, style)
    content["method"] = method
    S["content"], S["phase"] = content, "question"
    S["qcount"] += 1
    S["methods_tried"].append({"concept": concept["title"], "method": method})
    S["asked"].setdefault(concept["title"], []).append(content["question"]["text"])
    S["log"].append(f"Content agent wrote '{concept['title']}' (method: {method}, question: {kind})")


def begin_lesson(T):
    # Architect makes the plan, Content agent writes the first part. T is a working copy.
    keep = ("learner", "topic", "material", "profile", "quiz")
    T.update({k: v for k, v in new_state().items() if k not in keep})
    T["plan"] = make_lesson_plan(T)
    T["log"].append(f"Architect planned {len(T['plan'])} concepts")
    generate_content(T, "first_time")
    T["stage"] = "learning"


def process_answer(S, answer, skipped=False):
    # Called when the learner submits. Evaluate -> update counters -> Orchestrator decides.
    q = S["content"]["question"]
    concept = current_concept(S)
    if skipped:                                     # "I don't know" is useful information, not a mistake
        ev = {"result": "wrong", "right": "", "missing": "", "misconception": "", "skipped": True,
              "feedback": "That's okay. Not knowing is useful information for me, so I will help in a different way.",
              "follow_up": ""}
        shown = "(I don't know)"
    elif q["kind"] == "mcq":
        ev, shown = check_mcq(q, answer), q["options"][answer]
    else:
        ev, shown = evaluate_answer(S, q, answer), answer    # the only AI call here (may raise an error)
    S["attempts"] += 1
    if ev["result"] == "correct":
        S["wrong_streak"] = 0
    elif ev["result"] == "wrong":
        S["wrong_streak"] += 1
    action = decide_next(ev["result"], S)
    if action in ("worked_example", "skip_revision"):
        mark_revision(S, concept["title"])
    if is_final(S) and ev["result"] != "correct":
        mark_revision(S, "Applying all ideas (final challenge)")
    S["history"].append({"concept": concept["title"], "method": S["content"]["method"], "kind": q["kind"],
                         "answer": str(shown)[:200], "result": "skipped" if ev["skipped"] else ev["result"]})
    log_answer(S, concept["title"], q, shown, ev["result"])
    S["log"].append(f"Evaluator: {ev['result']}  ->  Orchestrator: {action}")
    S["last_eval"], S["action"], S["phase"] = ev, action, "feedback"
    save_progress(S)


def restore_after_review(S):
    # After checking the earlier idea, go back to the concept we came from
    S["current"], S["return_to"] = S["return_to"], None
    reset_counters(S)


def advance(S):
    # Called when the learner presses Continue. Works on a copy, so a model error never corrupts progress.
    T = copy.deepcopy(S)
    action, ev = T["action"], T["last_eval"]
    if action == "finish":
        T["stage"] = "done"
        commit(S, T)
        return
    title = current_concept(T)["title"]
    gap = " | ".join(x for x in [ev.get("missing"), ev.get("misconception"), ev.get("follow_up")] if x)
    if action in ("next_concept", "skip_revision"):
        if action == "next_concept" and title not in T["completed"]:
            T["completed"].append(title)
        if T["return_to"] is not None:               # we were only reviewing an earlier idea
            restore_after_review(T)
            generate_content(T, "after_review")
        else:
            T["current"] += 1
            reset_counters(T)
            generate_content(T, "first_time")        # becomes the final challenge after the last concept
    elif action == "go_back":
        T["return_to"], T["current"], T["attempts"] = T["current"], T["current"] - 1, 0
        generate_content(T, "go_back")
    elif action == "worked_example":
        T["rescue_used"] = True
        generate_content(T, "worked_example")
    elif action == "change_style":
        generate_content(T, "change_style", extra=gap, style=pick_style(T))
    else:                                            # hint_retry, fill_gap, follow_up
        generate_content(T, action, extra=gap)
    commit(S, T)

# ---------------- LOOK AND FEEL ----------------
# Design idea: a textbook page lying on dark graph paper. Ink-navy night background,
# one periwinkle accent, and a yellow highlighter used only to mark what the learner will learn.
# Every text colour is set here on purpose, so the page stays readable in any browser mode.
THEME_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Figtree:wght@400;500;600;700&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');

:root{
  --ink:#E9EDF9; --soft:#9EA9CB; --line:#2B3558; --paper:#0E1220; --grid:#151B2F;
  --card:#151B30; --field:#1B2240; --side:#0A0E1A;
  --cobalt:#7C97FF; --btn:#4F6EF7; --btn-d:#3A55D4; --tint:#1A2552; --marker:rgba(255,224,102,.42);
  --ok:#4FE0A0; --ok-bg:rgba(79,224,160,.12); --bad:#FF8091; --bad-bg:rgba(255,128,145,.12);
  --mid:#FFC857; --mid-bg:rgba(255,200,87,.12);
  --head:'Bricolage Grotesque','Figtree',system-ui,sans-serif;
  --body:'Figtree',system-ui,-apple-system,'Segoe UI',sans-serif;
  --read:'Literata',Georgia,'Times New Roman',serif;
  color-scheme: dark;
}

/* ---- page and text colours ---- */
.stApp{
  background-color:var(--paper);
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:32px 32px; color:var(--ink);
}
.stApp p,.stApp li,.stApp label,.stApp summary,.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.stApp [data-testid="stWidgetLabel"] p,.stApp [data-testid="stMarkdownContainer"] p{color:var(--ink);}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] p,.stApp small{color:var(--soft);}
.stApp a{color:var(--cobalt);}
.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button,.stApp summary{font-family:var(--body);}
.stApp h1,.stApp h2,.stApp h3{font-family:var(--head);letter-spacing:-.01em;}
.stApp code{font-size:.9em;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stElementContainer"]:has(style),.element-container:has(style){display:none;}
.block-container,[data-testid="stMainBlockContainer"]{max-width:900px;padding-top:2.4rem;padding-bottom:4rem;}
section[data-testid="stSidebar"]{background:var(--side);border-right:1px solid var(--line);color:var(--ink);}

/* ---- buttons ---- */
.stApp [data-testid^="stBaseButton-"]{
  border-radius:10px;font-weight:600;padding:.55rem 1.15rem;background:var(--field);color:var(--ink);
  border:1.5px solid #3C4878;transition:transform .08s,box-shadow .08s;}
.stApp [data-testid^="stBaseButton-"] p{color:var(--ink);}
.stApp [data-testid^="stBaseButton-"]:hover:not(:disabled){
  box-shadow:0 3px 0 #3C4878;transform:translateY(-1px);border-color:#6070AD;}
.stApp [data-testid="stBaseButton-primary"],.stApp [data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--btn);border-color:var(--btn-d);color:#fff;}
.stApp [data-testid="stBaseButton-primary"] p,.stApp [data-testid="stBaseButton-primaryFormSubmit"] p{color:#fff;}
.stApp [data-testid="stBaseButton-primary"]:hover:not(:disabled),.stApp [data-testid="stBaseButton-primaryFormSubmit"]:hover:not(:disabled){
  box-shadow:0 3px 0 var(--btn-d);border-color:var(--btn-d);}
.stApp [data-testid^="stBaseButton-"]:disabled{opacity:.45;}
.stApp button:focus-visible{outline:3px solid var(--cobalt);outline-offset:2px;}
section[data-testid="stSidebar"] [data-testid^="stBaseButton-"]{width:100%;}

/* ---- inputs ---- */
[data-testid="stTextInputRootElement"],[data-testid="stTextAreaRootElement"]{
  background:var(--field) !important;border:1.5px solid var(--line) !important;border-radius:10px !important;}
[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stTextAreaRootElement"]:focus-within{
  border-color:var(--cobalt) !important;box-shadow:0 0 0 3px rgba(124,151,255,.22);}
.stApp input,.stApp textarea{color:var(--ink) !important;caret-color:var(--ink);background:transparent !important;}
.stApp input::placeholder,.stApp textarea::placeholder{color:#7783A8 !important;opacity:1;}
[data-testid="stFileUploaderDropzone"]{background:var(--field) !important;border:1.5px dashed #4A5990 !important;border-radius:12px;}
[data-testid="stFileUploaderDropzone"],[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] span{color:var(--soft);}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:var(--card);}
[data-testid="stAlert"]{border-radius:12px;}
[role="radiogroup"]{gap:.5rem;}
[role="radiogroup"] label{
  background:var(--field);border:1.5px solid var(--line);border-radius:12px;padding:.6rem .9rem;
  transition:border-color .12s,background .12s;}
[role="radiogroup"] label:hover{border-color:var(--cobalt);}
[role="radiogroup"] label:has(input:checked){border-color:var(--cobalt);background:var(--tint);}
.st-key-practice_card [role="radiogroup"] label,.st-key-quiz_card [role="radiogroup"] label{width:100%;}

/* ---- cards (containers with a key) ---- */
.st-key-start_card,.st-key-profile_card,.st-key-practice_card,.st-key-quiz_card,.st-key-fb_card{
  background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1.4rem 1.5rem;}
.st-key-lesson_page{
  background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.7rem 1.9rem;
  box-shadow:5px 5px 0 #222C58;margin-bottom:1.2rem;}
.st-key-lesson_page [data-testid="stMarkdownContainer"] p,.st-key-lesson_page [data-testid="stMarkdownContainer"] li{
  font-family:var(--read);font-size:1.04rem;line-height:1.75;}
.st-key-example_box{background:#111833;border-left:4px solid var(--cobalt);border-radius:0 12px 12px 0;padding:.9rem 1.1rem;}
.st-key-fb_right{background:var(--ok-bg);border-left:4px solid var(--ok);border-radius:0 12px 12px 0;padding:.7rem 1rem;}
.st-key-fb_missing{background:var(--mid-bg);border-left:4px solid var(--mid);border-radius:0 12px 12px 0;padding:.7rem 1rem;}
.st-key-fb_right p,.st-key-fb_missing p{margin:0 0 .2rem;}

/* ---- default text colour for our own blocks (variants below override it) ---- */
.brand,.side-title,.plan-row,.hook,.callout,.rev,.stat b,.step b,.tbl td,.learner .l-name,.score b{color:var(--ink);}

/* ---- start screen ---- */
.stApp .hero h1{font-size:clamp(2rem,4.6vw,3.1rem);line-height:1.08;max-width:15em;margin:0 0 .8rem;font-weight:700;padding:0;}
.stApp .hero .lead{font-size:1.1rem;color:var(--soft);max-width:34em;line-height:1.6;margin:0 0 1.6rem;}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;margin:0 0 1.8rem;}
.step{display:flex;gap:.7rem;align-items:flex-start;}
.step .n{flex:none;width:1.9rem;height:1.9rem;border-radius:50%;background:var(--ink);color:var(--paper);
  font-family:var(--head);font-weight:700;font-size:.9rem;display:grid;place-items:center;}
.step b{display:block;font-family:var(--head);font-size:1rem;margin-bottom:.15rem;}
.step span.t{display:block;color:var(--soft);font-size:.92rem;line-height:1.45;}
.stApp .page-head h1{font-size:2.1rem;line-height:1.12;margin:0 0 .3rem;padding:0;}
.stApp .page-head p{color:var(--soft);margin:0 0 1.2rem;font-size:1.02rem;}

/* ---- sidebar pieces ---- */
.brand{font-family:var(--head);font-weight:700;font-size:1.45rem;letter-spacing:-.01em;margin:0;}
.stApp .brand-sub{color:var(--soft);font-size:.88rem;margin:0 0 1.1rem;}
.side-title{font-weight:700;font-size:.92rem;margin:0 0 .35rem;}
.key-state{font-size:.84rem;margin:.3rem 0 .3rem;padding:.3rem .6rem;border-radius:8px;}
.key-state.ok{background:var(--ok-bg);color:var(--ok);}
.key-state.no{background:var(--mid-bg);color:var(--mid);}
.stApp .side-link{font-size:.84rem;color:var(--cobalt) !important;}
.side-rule{border:0;border-top:1px solid var(--line);margin:1rem 0;}
.learner .l-name{font-family:var(--head);font-weight:700;font-size:1.1rem;}
.learner .l-topic{color:var(--soft);font-size:.92rem;margin:.1rem 0 .5rem;}
.chip{display:inline-block;padding:.12rem .6rem;border-radius:999px;background:var(--field);border:1px solid var(--line);
  font-size:.8rem;color:var(--soft);margin:0 .3rem .3rem 0;}
.plan-title{font-weight:700;font-size:.92rem;margin:.9rem 0 .3rem;color:var(--ink);}
.plan-row{display:flex;gap:.6rem;align-items:flex-start;padding:.3rem 0;font-size:.92rem;line-height:1.35;}
.plan-row .dot{flex:none;width:1.45rem;height:1.45rem;border-radius:50%;display:grid;place-items:center;
  font-weight:700;font-size:.74rem;border:1.5px solid #4A5683;color:var(--soft);background:var(--field);}
.plan-row.done .dot{background:var(--ok);border-color:var(--ok);color:var(--paper);}
.plan-row.now .dot{background:var(--cobalt);border-color:var(--cobalt);color:var(--paper);}
.plan-row.now .pt{font-weight:700;}
.plan-row.revise .dot{background:var(--mid-bg);border-color:var(--mid);color:var(--mid);}
.score{margin:.8rem 0 .2rem;font-size:.92rem;color:var(--soft);}
.score b{font-family:var(--head);font-size:1.5rem;}

/* ---- learning screen ---- */
.pwrap{display:flex;align-items:center;gap:.8rem;margin:0 0 1.1rem;color:var(--soft);font-size:.9rem;}
.pbar{flex:1;height:8px;background:#232C4F;border-radius:99px;overflow:hidden;}
.pbar i{display:block;height:100%;background:var(--cobalt);border-radius:99px;}
.concept-head .where{color:var(--soft);font-size:.92rem;font-weight:600;}
.stApp .concept-head h2{margin:.1rem 0 1rem;font-size:2rem;line-height:1.15;padding:0;}
.gagne{display:inline-block;font-family:var(--body) !important;font-weight:600;font-size:.8rem !important;line-height:1.3 !important;
  color:var(--soft);background:#1C2445;border-radius:8px;padding:.2rem .6rem;margin:0 0 .8rem;}
.hook{font-family:var(--read);font-size:1.12rem;line-height:1.6;border-left:4px solid var(--cobalt);padding:.1rem 0 .1rem 1rem;margin:0 0 1rem;}
.goal{margin:0 0 .5rem;}
.goal mark{background:linear-gradient(transparent 52%,var(--marker) 52%);color:inherit;padding:0 .15em;
  -webkit-box-decoration-break:clone;box-decoration-break:clone;}
.stApp .recall{color:var(--soft);margin:0 0 1rem;}
.callout{border-left:4px solid var(--cobalt);background:var(--tint);border-radius:0 12px 12px 0;padding:.8rem 1rem;margin:.8rem 0;line-height:1.5;}
.callout.warn{border-color:var(--mid);background:var(--mid-bg);}
.src{font-size:.88rem;margin:.2rem 0 .8rem;}
.src.ok{color:var(--ok);} .src.warn{color:var(--mid);}
.verdict{display:flex;align-items:center;gap:.7rem;border-radius:14px;padding:.8rem 1.1rem;
  font-family:var(--head);font-weight:700;font-size:1.35rem;margin:0 0 1rem;}
.verdict .mk{flex:none;width:2rem;height:2rem;border-radius:50%;display:grid;place-items:center;color:var(--paper);font-size:1rem;}
.verdict.ok{background:var(--ok-bg);color:var(--ok);} .verdict.ok .mk{background:var(--ok);}
.verdict.bad{background:var(--bad-bg);color:var(--bad);} .verdict.bad .mk{background:var(--bad);}
.verdict.mid{background:var(--mid-bg);color:var(--mid);} .verdict.mid .mk{background:var(--mid);}
.verdict.info{background:var(--tint);color:var(--cobalt);} .verdict.info .mk{background:var(--cobalt);}

/* ---- summary screen ---- */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0 1.6rem;}
.stat{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1rem 1.2rem;}
.stat b{display:block;font-family:var(--head);font-size:2.2rem;line-height:1.1;}
.stat span{color:var(--soft);font-size:.9rem;}
.tbl-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:var(--card);margin:0 0 1.4rem;}
.tbl{width:100%;border-collapse:collapse;font-size:.92rem;background:transparent;}
.tbl th{text-align:left;font-weight:600;color:var(--soft);padding:.6rem .9rem;border-bottom:1px solid var(--line);background:#1A2140;}
.tbl td{padding:.6rem .9rem;border-bottom:1px solid #222B4D;background:transparent;}
.tbl tr:last-child td{border-bottom:0;}
.badge{display:inline-block;padding:.1rem .6rem;border-radius:999px;font-size:.8rem;font-weight:600;}
.badge.ok{background:var(--ok-bg);color:var(--ok);} .badge.bad{background:var(--bad-bg);color:var(--bad);}
.badge.mid{background:var(--mid-bg);color:var(--mid);} .badge.info{background:var(--tint);color:var(--cobalt);}
.rev{padding:.6rem 1rem;background:var(--mid-bg);border-left:4px solid var(--mid);border-radius:0 12px 12px 0;margin:0 0 .5rem;}
.rev small{color:var(--soft);}

@media (max-width:700px){
  .steps,.stats{grid-template-columns:1fr;}
  .st-key-lesson_page{padding:1.2rem 1.1rem;}
  .stApp .concept-head h2{font-size:1.6rem;}
}
@media (prefers-reduced-motion:reduce){*{transition:none !important;}}
</style>"""

# ---------------- STREAMLIT: SMALL HELPERS AND SIDEBAR ----------------
def esc(text):
    # Make AI-written text safe to place inside our own HTML (also removes line breaks)
    return html.escape(" ".join(str(text).split()), quote=True)


def page_head(title, sub=""):
    st.markdown(f'<div class="page-head"><h1>{esc(title)}</h1>' + (f"<p>{esc(sub)}</p>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def gagne(text):
    # A small tag that shows which of Gagne's nine events the learner is looking at
    st.markdown(f'<span class="gagne">{esc(text)}</span>', unsafe_allow_html=True)


def run_with_spinner(message, fn, *args):
    # Run a slow step with a spinner. If the AI fails, show a friendly error (nothing is lost).
    try:
        with st.spinner(message):
            fn(*args)
        return True
    except Exception as e:
        st.error(f"Something went wrong: {e}\n\nYour progress is safe. Press the button again.")
        return False


def key_box():
    # The visitor pastes their own Groq key here. It lives only in this browser session.
    st.markdown('<div class="side-title">Groq API key</div>', unsafe_allow_html=True)
    st.text_input("Groq API key", type="password", key="groq_key", placeholder="Paste your key, then press Enter",
                  label_visibility="collapsed")
    key = current_key()
    if not key:
        state = '<div class="key-state no">No key yet. Paste it above to start.</div>'
    elif not key.startswith("gsk_"):
        state = '<div class="key-state no">Groq keys start with gsk_. Check what you pasted.</div>'
    else:
        state = '<div class="key-state ok">Key added for this session.</div>'
    st.markdown(state + '<a class="side-link" href="https://console.groq.com/keys" target="_blank" '
                        'rel="noopener noreferrer">Get a free Groq key</a>', unsafe_allow_html=True)


def sidebar(S):
    with st.sidebar:
        st.markdown('<div class="brand">AI Teacher</div>'
                    '<div class="brand-sub">Lessons that adapt when you get stuck</div>', unsafe_allow_html=True)
        key_box()
        st.markdown('<hr class="side-rule">', unsafe_allow_html=True)

        if S["learner"] or S["topic"]:
            chips = ""
            if S["profile"]:
                p = S["profile"]
                chips = f'<span class="chip">{esc(p["level"])}</span><span class="chip">goal: {esc(p["goal"])}</span>'
            st.markdown(f'<div class="learner"><div class="l-name">{esc(S["learner"] or "Learner")}</div>'
                        f'<div class="l-topic">{esc(S["topic"])}</div>{chips}</div>', unsafe_allow_html=True)
            if S["profile"] and S["profile"].get("level_note"):
                st.caption(f"Level {S['profile']['level_note']}")

        if S["plan"]:
            revision = [r["concept"] for r in S["needs_revision"]]
            rows = ['<div class="plan-title">Lesson plan</div>']
            for i, c in enumerate(S["plan"]):
                t = c["title"]
                if t in revision:
                    cls, mark = "revise", "&#8635;"
                elif t in S["completed"]:
                    cls, mark = "done", "&#10003;"
                elif S["stage"] == "learning" and i == S["current"]:
                    cls, mark = "now", str(i + 1)
                else:
                    cls, mark = "todo", str(i + 1)
                rows.append(f'<div class="plan-row {cls}"><span class="dot">{mark}</span><span class="pt">{esc(t)}</span></div>')
            if S["stage"] == "done":
                cls = "done"
            elif S["stage"] == "learning" and is_final(S):
                cls = "now"
            else:
                cls = "todo"
            rows.append(f'<div class="plan-row {cls}"><span class="dot">&#9733;</span><span class="pt">Final challenge</span></div>')
            st.markdown("".join(rows), unsafe_allow_html=True)

        if S["history"]:
            right = sum(1 for h in S["history"] if h["result"] == "correct")
            st.markdown(f'<div class="score"><b>{right}</b> of {len(S["history"])} answers correct</div>', unsafe_allow_html=True)

        if S["log"]:
            with st.expander("Agent activity log"):
                for line in reversed(S["log"][-15:]):
                    st.caption(line)

        if S["stage"] in ("learning", "done", "quiz") or S["learner"]:
            st.markdown('<hr class="side-rule">', unsafe_allow_html=True)
        if S["stage"] in ("learning", "done", "quiz"):
            if st.button("Change profile and restart"):
                keep = {k: S[k] for k in ("learner", "topic", "material", "profile")}
                fresh = new_state()
                fresh.update(keep)
                fresh["stage"] = "profile"
                commit(S, fresh)
                st.rerun()
        if S["learner"]:
            sure = st.checkbox("I want to delete my saved data")
            if st.button("Delete my saved data", disabled=not sure):
                delete_learner(S["learner"])
                S.clear()
                S.update(new_state())
                st.rerun()

# ---------------- STREAMLIT SCREENS 1-3: START, PROFILE, QUICK CHECK ----------------
HERO_HTML = (
    '<div class="hero">'
    "<h1>A teacher that changes its method when you get stuck.</h1>"
    '<p class="lead">Pick a topic or upload your own notes. The lesson is planned for your level, '
    "one idea is taught at a time, and every answer is checked.</p>"
    '<div class="steps">'
    '<div class="step"><span class="n">1</span><div><b>Tell it about you</b>'
    '<span class="t">Your level, your goal and how you like to learn.</span></div></div>'
    '<div class="step"><span class="n">2</span><div><b>Learn and answer</b>'
    '<span class="t">One idea, one question, feedback straight away.</span></div></div>'
    '<div class="step"><span class="n">3</span><div><b>Stuck? It adapts</b>'
    '<span class="t">Hints, simpler words, then a worked example.</span></div></div>'
    "</div></div>"
)


def screen_start(S):
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    has_key = bool(current_key())
    if not has_key:
        st.markdown('<div class="callout warn">Add your Groq API key in the sidebar to begin. '
                    'Keys are free at <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer">'
                    "console.groq.com/keys</a>.</div>", unsafe_allow_html=True)
    go = False
    with st.container(key="start_card"):
        name = st.text_input("Your name", value=S.get("learner", ""), placeholder="Used to save your progress")
        saved = load_progress(name.strip()) if name.strip() else None
        if saved and saved.get("topic"):
            st.markdown(f'<div class="callout">Saved lesson found: <b>{esc(saved["topic"])}</b></div>', unsafe_allow_html=True)
            if st.button("Resume this lesson"):
                S.clear()
                S.update(saved)
                st.rerun()
        topic = st.text_input("What do you want to learn?", placeholder="For example: Python loops, or machine learning basics")
        pdf = st.file_uploader("Or upload your notes as a PDF (optional)", type=["pdf"])
        go = st.button("Start lesson", type="primary", disabled=not has_key)
    if go:
        if not name.strip():
            st.warning("Type your name first.")
            return
        if not topic.strip() and pdf is None:
            st.warning("Type a topic or upload a PDF.")
            return
        material = None
        if pdf is not None:
            try:
                with st.spinner("Reading your PDF..."):
                    material = read_pdf(pdf.getvalue(), pdf.name)
            except Exception as e:
                st.error(f"I could not read this PDF ({e}). Try another file or type a topic.")
                return
        fresh = new_state()
        fresh.update(learner=name.strip(), material=material, stage="profile",
                     topic=topic.strip() or os.path.splitext(pdf.name)[0])
        commit(S, fresh)
        st.rerun()


def start_after_profile(S, profile, take_quiz):
    # Work on a copy T: if the AI fails, the real state S stays unchanged.
    T = copy.deepcopy(S)
    T["profile"] = profile
    if take_quiz:
        T["quiz"] = make_quiz(T)
        if T["quiz"]:
            T["stage"] = "quiz"
            commit(S, T)
            return
    begin_lesson(T)
    commit(S, T)


def screen_profile(S):
    page_head("Set up your lesson", "Three quick choices. Skip them and I will use sensible defaults.")
    old = S.get("profile") or default_profile()
    with st.container(key="profile_card"):
        level = st.radio("Your level in this topic", LEVELS, horizontal=True,
                         index=LEVELS.index(old["level"]) if old.get("level") in LEVELS else 0)
        goal = st.selectbox("Why are you learning this?", GOALS,
                            index=GOALS.index(old["goal"]) if old.get("goal") in GOALS else 3)
        style = st.selectbox("How do you like to learn?", STYLE_PREFS,
                             index=STYLE_PREFS.index(old["style"]) if old.get("style") in STYLE_PREFS else 1)
        quiz = st.checkbox("Start with a 3-question quick check (optional)")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Continue", type="primary"):
        if run_with_spinner("The agents are planning your lesson...", start_after_profile,
                            S, {"level": level, "goal": goal, "style": style}, quiz):
            st.rerun()
    if c2.button("Skip questions"):
        if run_with_spinner("The agents are planning your lesson...", start_after_profile,
                            S, default_profile(), False):
            st.rerun()


def finish_quiz(S, score):
    T = copy.deepcopy(S)
    apply_quiz(T, score)
    begin_lesson(T)
    commit(S, T)


def screen_quiz(S):
    page_head("Quick check", "Three short questions so I can start at the right level.")
    with st.container(key="quiz_card"):
        with st.form("quiz_form", border=False):
            picks = [st.radio(f"{i + 1}. {q['text']}", q["options"], index=None, key=f"quiz_{i}")
                     for i, q in enumerate(S["quiz"])]
            submitted = st.form_submit_button("Submit quick check", type="primary")
    if submitted:
        score = sum(1 for q, pick in zip(S["quiz"], picks)
                    if pick is not None and q["options"].index(pick) == q["correct_index"])
        if run_with_spinner("Planning your lesson...", finish_quiz, S, score):
            st.rerun()
    if st.button("Skip the quick check"):
        if run_with_spinner("Planning your lesson...", finish_quiz, S, len(S["quiz"])):
            st.rerun()

# ---------------- STREAMLIT SCREENS 4-5: LEARNING SCREEN AND SUMMARY ----------------
def show_lesson(S, concept, c):
    # Gagne's events are shown as small tags so the structure of the lesson is visible.
    with st.container(key="lesson_page"):
        if c["method"] == "first_time":
            gagne("Gagné steps 1 to 3: gain attention, state the goal, recall prior knowledge")
            if concept.get("hook"):
                st.markdown(f'<div class="hook">{esc(concept["hook"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="goal"><b>You will learn:</b> <mark>{esc(concept.get("goal", ""))}</mark></p>',
                        unsafe_allow_html=True)
            if concept.get("recall"):
                st.markdown(f'<p class="recall"><b>You may already know:</b> {esc(concept["recall"])}</p>',
                            unsafe_allow_html=True)
        elif c["method"] != "final":
            st.markdown('<div class="callout warn"><b>New approach from the tutor:</b> '
                        f'{esc(METHOD_LABEL.get(c["method"], c["method"]))}</div>', unsafe_allow_html=True)
        gagne("Gagné step 4: present the new idea" if c["method"] != "final"
              else "Gagné steps 8 and 9: test what you learned and use it in real life")
        st.markdown(c["explanation"])
        if c.get("example"):
            with st.container(key="example_box"):
                st.markdown("**Example**")
                st.markdown(c["example"])
    if c["source_status"] == "verified":
        st.markdown('<div class="src ok">&#10003; This explanation is supported by your PDF.</div>', unsafe_allow_html=True)
    elif c["source_status"] == "unverified":
        st.markdown('<div class="src warn">I could not match this explanation to a sentence in your PDF. '
                    "Please double-check it.</div>", unsafe_allow_html=True)
    if c.get("hint"):
        with st.expander("Need a hint?"):
            st.write(c["hint"])


def show_question(S, q):
    key = f"ans_{S['qcount']}"
    with st.container(key="practice_card"):
        gagne("Gagné steps 5 and 6: guidance and practice")
        if q["kind"] == "mcq":
            choice = st.radio(q["text"], q["options"], index=None, key=key)
            ready = choice is not None
            answer = q["options"].index(choice) if ready else None
        else:
            answer = st.text_area(q["text"], key=key, height=150, placeholder="Write the answer in your own words...")
            ready = bool(answer.strip())
        c1, c2, _ = st.columns([1, 1, 1.4])
        submit = c1.button("Submit answer", type="primary", disabled=not ready)
        unsure = c2.button("I don't know")
    if submit:
        if run_with_spinner("The Evaluator is checking your answer...", process_answer, S, answer):
            st.rerun()
    if unsure:
        if run_with_spinner("Thinking about how to help...", process_answer, S, None, True):
            st.rerun()


def show_feedback(S):
    ev, q = S["last_eval"], S["content"]["question"]
    if ev["skipped"]:
        kind, mark, word = "info", "&bull;", "No problem"
    elif ev["result"] == "correct":
        kind, mark, word = "ok", "&#10003;", "Correct"
    elif ev["result"] == "partial":
        kind, mark, word = "mid", "&frac12;", "Partly correct"
    elif ev["result"] == "unclear":
        kind, mark, word = "info", "?", "I need a little more detail"
    else:
        kind, mark, word = "bad", "&#10005;", "Not quite"
    gagne("Gagné step 7: feedback")
    st.markdown(f'<div class="verdict {kind}"><span class="mk">{mark}</span><span>{word}</span></div>', unsafe_allow_html=True)
    with st.container(key="fb_card"):
        st.markdown(ev["feedback"])
        if ev.get("right"):
            with st.container(key="fb_right"):
                st.markdown(f"**What you got right**\n\n{ev['right']}")
        if ev.get("missing"):
            with st.container(key="fb_missing"):
                st.markdown(f"**What is missing**\n\n{ev['missing']}")
        if q["kind"] == "mcq":
            st.markdown(f"**Correct answer:** {q['options'][q['correct_index']]}")
        else:
            with st.expander("See a model answer"):
                st.write(q["sample_answer"])
    st.markdown(f'<div class="callout"><b>Tutor\'s next move:</b> {esc(ACTION_TEXT[S["action"]])}</div>', unsafe_allow_html=True)
    if st.button("Continue", type="primary"):
        if run_with_spinner("The agents are preparing the next step...", advance, S):
            st.rerun()


def screen_learning(S):
    if not S.get("content"):
        st.error("This lesson has no content yet. Use 'Change profile and restart' in the sidebar.")
        return
    n = len(S["plan"])
    concept, c, q = current_concept(S), S["content"], S["content"]["question"]
    done = min(len(S["completed"]), n + 1)
    st.markdown(f'<div class="pwrap"><div class="pbar"><i style="width:{int(100 * done / (n + 1))}%"></i></div>'
                f"<span>{done} of {n + 1} steps done</span></div>", unsafe_allow_html=True)
    where = "Final challenge" if is_final(S) else f"Concept {S['current'] + 1} of {n}"
    title = concept["title"].replace("Final challenge: ", "") if is_final(S) else concept["title"]
    st.markdown(f'<div class="concept-head"><span class="where">{esc(where)}</span><h2>{esc(title)}</h2></div>',
                unsafe_allow_html=True)
    if S["return_to"] is not None:
        st.caption("Reviewing an earlier idea first. After that we return to the concept we came from.")
    if S["phase"] == "question":
        show_lesson(S, concept, c)
        show_question(S, q)
    else:
        show_feedback(S)


def screen_done(S):
    h = S["history"]
    right = sum(1 for x in h if x["result"] == "correct")
    changed = sum(1 for x in h if x["method"] not in ("first_time", "final"))
    page_head("Lesson finished", S["topic"])
    st.markdown(f'<div class="stats"><div class="stat"><b>{len(h)}</b><span>Questions answered</span></div>'
                f'<div class="stat"><b>{right}</b><span>Correct</span></div>'
                f'<div class="stat"><b>{changed}</b><span>Times the teacher changed method</span></div></div>',
                unsafe_allow_html=True)
    st.subheader("Your answers")
    if h:
        badge = {"correct": ("ok", "Correct"), "wrong": ("bad", "Wrong"), "partial": ("mid", "Partly"),
                 "unclear": ("info", "Unclear"), "skipped": ("info", "Skipped")}
        rows = ""
        for x in h:
            cls, label = badge.get(x["result"], ("info", x["result"]))
            rows += (f'<tr><td>{esc(x["concept"])}</td><td>{esc(x["method"].replace("_", " "))}</td>'
                     f'<td>{"multiple choice" if x["kind"] == "mcq" else "written"}</td>'
                     f'<td><span class="badge {cls}">{label}</span></td></tr>')
        st.markdown('<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Concept</th><th>Method</th><th>Type</th>'
                    f"<th>Result</th></tr></thead><tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)
    st.subheader("Topics that need revision")
    if S["needs_revision"]:
        for item in S["needs_revision"]:
            tried = ", ".join(m.replace("_", " ") for m in item["methods_tried"])
            st.markdown(f'<div class="rev"><b>{esc(item["concept"])}</b><br><small>Methods already tried: {esc(tried)}</small></div>',
                        unsafe_allow_html=True)
        st.caption("These topics are saved. If they stay difficult, review them with a teacher.")
    else:
        st.success("Nothing needs revision. Well done!")
    if st.button("Start another lesson", type="primary"):
        name = S["learner"]
        fresh = new_state()
        fresh["learner"] = name
        commit(S, fresh)
        st.rerun()

# ---------------- MAIN: ROUTES THE CURRENT STAGE TO ITS SCREEN ----------------
def main():
    st.set_page_config(page_title="AI Teacher - Algorithmic Instructional Designer", page_icon="🎓",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    init_db()
    if "S" not in st.session_state:
        st.session_state["S"] = new_state()
    S = st.session_state["S"]
    sidebar(S)
    if S["stage"] != "start" and not current_key():
        st.markdown('<div class="callout warn">Your Groq API key is missing. Add it in the sidebar to continue.</div>',
                    unsafe_allow_html=True)
    screens = {"start": screen_start, "profile": screen_profile, "quiz": screen_quiz,
               "learning": screen_learning, "done": screen_done}
    screens.get(S["stage"], screen_start)(S)


if __name__ == "__main__":      # Streamlit runs this file as __main__
    main()
