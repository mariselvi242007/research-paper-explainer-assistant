import html
import re
import uuid
import hashlib
from io import BytesIO
from urllib.parse import quote
from datetime import datetime, timedelta, timezone

import streamlit as st
import chromadb
from groq import Groq
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_MODEL = "gemini-3-flash-preview"

GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

PREFERRED_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

DISPLAY_TZ = timezone(timedelta(hours=5, minutes=30), "IST")

MODES = ["Simple", "Technical"]

MODE_HINTS = {
    "Simple": "Plain, beginner-friendly answers.",
    "Technical": "Detailed answers with research terminology.",
}

STYLE_INSTRUCTIONS = {
    "Simple": (
        "Explain the answer in simple English. Use beginner-friendly "
        "language and avoid unnecessary technical terminology."
    ),
    "Technical": (
        "Give a technical and detailed explanation. Use appropriate "
        "research and computer science terminology."
    ),
}

BLUE = "#2F5D8A"
BLUE_DARK = "#244A6E"
BLUE_SOFT = "#E4EDF8"


# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@400;600&family=Public+Sans:wght@400;500;600;700&display=swap');

.stApp {{
    font-family: 'Public Sans', system-ui, sans-serif;
}}

/* ---- Hide Streamlit's own top-right toolbar and footer ---- */
footer,
#MainMenu,
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
.stDeployButton {{
    display: none !important;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

.block-container {{
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}}

/* ---- Typography ---- */
.pl-logo {{
    font: 600 1.7rem 'Newsreader', Georgia, serif;
    margin-bottom: .4rem;
    color: {BLUE};
}}

.pl-section {{
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #7A8899;
    margin: 1.2rem 0 .4rem 0;
}}

.pl-title {{
    font: 600 1.65rem 'Newsreader', Georgia, serif;
    line-height: 1.3;
    margin-bottom: .2rem;
}}

.pl-meta {{
    color: #66758a;
    font-size: .85rem;
    margin-bottom: .6rem;
}}

.pl-hero-title {{
    font: 600 2.8rem 'Newsreader', Georgia, serif;
    line-height: 1.15;
    margin: 2rem 0 .6rem 0;
}}

.pl-hero-sub {{
    color: #66758a;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}}

.pl-empty {{
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
}}

.pl-empty b {{
    font: 600 1.5rem 'Newsreader', Georgia, serif;
}}

.pl-empty span {{
    display: block;
    color: #66758a;
    margin-top: .4rem;
}}

.pl-cites {{
    color: #66758a;
    font-size: .82rem;
    margin-top: .5rem;
}}

/* ---- Chat layout: question on the right, answer on the left ---- */
.pl-user-row {{
    display: flex;
    justify-content: flex-end;
    margin: 1rem 0 .6rem 0;
}}

.pl-user-bubble {{
    background: {BLUE_SOFT};
    color: #1B2B3C;
    padding: .75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    line-height: 1.5;
    word-wrap: break-word;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px;
}}

/* ---- Inputs ---- */
[data-testid="stChatInput"]:focus-within {{
    border-color: {BLUE} !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed #9DB6CF;
    border-radius: 16px;
    padding: 1.5rem;
    background: #F7FAFD;
}}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: #F7F9FC;
}}

[data-testid="stSidebar"] button {{
    border-radius: 8px;
}}

.stButton button {{
    transition: all .15s ease;
}}

/* ---- Primary buttons: blue instead of red ---- */
button[kind="primary"],
[data-testid="stBaseButton-primary"] {{
    background-color: {BLUE} !important;
    border-color: {BLUE} !important;
    color: #FFFFFF !important;
}}

button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {{
    background-color: {BLUE_DARK} !important;
    border-color: {BLUE_DARK} !important;
    color: #FFFFFF !important;
}}

button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {{
    border-color: {BLUE} !important;
    color: {BLUE} !important;
}}

/* ---- Chat lists (Favourites and Recents) ---- */
[class*="st-key-chatlist"] button {{
    justify-content: flex-start !important;
    text-align: left !important;
}}

[class*="st-key-chatlist"] button[kind="secondary"],
[class*="st-key-chatlist"] [data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #33475B !important;
}}

[class*="st-key-chatlist"] button[kind="secondary"]:hover,
[class*="st-key-chatlist"] [data-testid="stBaseButton-secondary"]:hover {{
    background: #EAF0F7 !important;
    border-color: transparent !important;
    color: {BLUE} !important;
}}

[class*="st-key-chatlist"] button[kind="primary"],
[class*="st-key-chatlist"] [data-testid="stBaseButton-primary"] {{
    background: {BLUE_SOFT} !important;
    border: 1px solid transparent !important;
    border-left: 3px solid {BLUE} !important;
    color: #1F4468 !important;
    font-weight: 600;
}}

/* ---- Chat actions: quiet, text-style buttons ---- */
.st-key-chat_actions button {{
    justify-content: flex-start !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    color: #33475B !important;
    font-weight: 500;
}}

.st-key-chat_actions button:hover {{
    background: #EAF0F7 !important;
    color: {BLUE} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

defaults = {
    "current_chat_id": None,
    "chats": {},
    "recents": [],
    "pinned": [],
    "explanation_mode": "Simple",
    "upload_key": 0,
    "add_key": 0,
    "pending": None,
    "confirm_delete": None,
    "notice": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 4. API CLIENTS
#     PRIORITY: GROQ -> GEMINI
# ============================================================

groq_client = None
groq_model = None
client = None


def get_secret(name, default=""):
    """Safely retrieve a value from Streamlit Secrets."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# Initialize Groq first
try:
    groq_key = get_secret("GROQ_API_KEY")

    if groq_key:
        groq_client = Groq(api_key=groq_key)

        available_models = groq_client.models.list()
        available_ids = [
            model.id for model in available_models.data
        ]

        for candidate in PREFERRED_GROQ_MODELS:
            if candidate in available_ids:
                groq_model = candidate
                break

        if groq_model is None:
            for model_id in available_ids:
                if any(
                    name in model_id.lower()
                    for name in ["llama", "gpt-oss", "qwen"]
                ):
                    groq_model = model_id
                    break

except Exception as e:
    groq_client = None
    groq_model = None
    print("Groq initialization failed:", str(e)[:300])


# Initialize Gemini as the fallback
try:
    gemini_key = get_secret("GEMINI_API_KEY")

    if gemini_key:
        client = genai.Client(api_key=gemini_key)

except Exception as e:
    client = None
    print("Gemini initialization failed:", str(e)[:300])


# Backend details are no longer shown in the interface.
if not (
    groq_client is not None and groq_model is not None
) and client is None:
    st.error(
        "No AI provider is available. Add GROQ_API_KEY and/or "
        "GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()


# ============================================================
# 5. EMBEDDING MODEL AND CHROMADB
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma_client():
    return chromadb.Client()


embedding_model = load_embedding_model()
chroma_client = load_chroma_client()


# ============================================================
# 6. PDF PROCESSING
# ============================================================

def extract_pdf_pages(file_bytes):
    """Extract text from each PDF page."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def create_chunks(pages):
    """Split the paper into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = []
    page_numbers = []

    for page_data in pages:
        page_chunks = splitter.split_text(page_data["text"])

        for chunk in page_chunks:
            chunks.append(chunk)
            page_numbers.append(page_data["page"])

    return chunks, page_numbers


def create_collection_name(file_bytes):
    """Create a stable collection name for the PDF."""
    digest = hashlib.md5(file_bytes).hexdigest()[:10]
    return f"research_paper_{digest}"


def process_pdf(file_bytes, paper_name):
    """Extract, embed, and index one uploaded PDF."""
    pages = extract_pdf_pages(file_bytes)

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF. "
            "Scanned PDFs may require OCR."
        )

    chunks, page_numbers = create_chunks(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    collection_name = create_collection_name(file_bytes)

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    if collection.count() == 0:
        embeddings = embedding_model.encode(
            chunks,
            show_progress_bar=False,
        )

        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[
                {"page": page}
                for page in page_numbers
            ],
        )

    return {
        "id": collection_name,
        "name": paper_name,
        "pages": pages,
        "collection": collection,
    }


def process_uploaded_files(files):
    """Process several PDFs. Returns (papers, error_messages)."""
    papers = []
    errors = []

    for uploaded in files:
        try:
            papers.append(
                process_pdf(uploaded.getvalue(), uploaded.name)
            )
        except Exception as e:
            errors.append(f"{uploaded.name}: {e}")

    return papers, errors


# ============================================================
# 7. CHAT MANAGEMENT
# ============================================================

def make_title(papers):
    if not papers:
        return "New chat"

    if len(papers) == 1:
        return papers[0]["name"]

    return f"{papers[0]['name']} +{len(papers) - 1} more"


def create_chat(papers):
    chat_id = uuid.uuid4().hex[:10]

    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "title": make_title(papers),
        "papers": papers,
        "history": [],
    }

    st.session_state.recents.insert(0, chat_id)
    st.session_state.current_chat_id = chat_id
    st.session_state.pending = None

    return chat_id


def add_papers_to_chat(chat, new_papers):
    """Add papers to an existing chat, skipping duplicates."""
    existing = {paper["id"] for paper in chat["papers"]}
    added = 0

    for paper in new_papers:
        if paper["id"] not in existing:
            chat["papers"].append(paper)
            existing.add(paper["id"])
            added += 1

    chat["title"] = make_title(chat["papers"])

    return added


def get_current_chat():
    chat_id = st.session_state.current_chat_id

    if chat_id:
        return st.session_state.chats.get(chat_id)

    return None


def move_to_top(chat_id):
    if chat_id in st.session_state.recents:
        st.session_state.recents.remove(chat_id)

    st.session_state.recents.insert(0, chat_id)


def open_chat(chat_id):
    if chat_id not in st.session_state.chats:
        return

    st.session_state.current_chat_id = chat_id
    st.session_state.pending = None


def new_chat():
    st.session_state.current_chat_id = None
    st.session_state.pending = None
    st.session_state.upload_key += 1


def delete_chat(chat_id):
    st.session_state.chats.pop(chat_id, None)

    if chat_id in st.session_state.recents:
        st.session_state.recents.remove(chat_id)

    if chat_id in st.session_state.pinned:
        st.session_state.pinned.remove(chat_id)

    if st.session_state.current_chat_id == chat_id:
        new_chat()


def set_mode(mode_name):
    st.session_state.explanation_mode = mode_name


def set_confirm_delete(chat_id):
    st.session_state.confirm_delete = chat_id


def clear_chat(chat_id):
    chat = st.session_state.chats.get(chat_id)

    if chat:
        chat["history"] = []


def toggle_pin(chat_id):
    if chat_id in st.session_state.pinned:
        st.session_state.pinned.remove(chat_id)
    else:
        st.session_state.pinned.insert(0, chat_id)


def ask_suggestion(question):
    st.session_state.pending = {
        "kind": "question",
        "question": question,
    }


def ask_summary(multi=False):
    st.session_state.pending = {
        "kind": "summary",
        "question": (
            "Summarize these papers and explain how they relate."
            if multi
            else "Summarize this paper."
        ),
    }


# ============================================================
# 8. AI ERROR HANDLING
# ============================================================

class QuotaExceeded(Exception):
    """Raised when Gemini models are exhausted or unavailable."""

    def __init__(
        self,
        quota_models,
        unavailable_models,
        retry_seconds=None,
    ):
        super().__init__("Gemini quota exceeded")
        self.quota_models = quota_models
        self.unavailable_models = unavailable_models
        self.retry_seconds = retry_seconds


def parse_retry_seconds(message):
    """Extract a retry delay from a model API error."""
    match = re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
        message,
    )

    if match:
        return float(match.group(1))

    match = re.search(
        r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*"
        r"(?:(\d+(?:\.\d+)?)s)?",
        message,
    )

    if match and any(match.groups()):
        hours, minutes, seconds = match.groups()

        return (
            int(hours or 0) * 3600
            + int(minutes or 0) * 60
            + float(seconds or 0)
        )

    return None


def get_model_candidates():
    """Return the Gemini model order."""
    primary = get_secret("GEMINI_MODEL", PRIMARY_MODEL)

    return [primary] + [
        model
        for model in GEMINI_FALLBACK_MODELS
        if model != primary
    ]


def format_wait(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60

    parts = []

    if hours:
        parts.append(
            f"{hours} hour{'s' if hours != 1 else ''}"
        )

    if minutes:
        parts.append(
            f"{minutes} minute{'s' if minutes != 1 else ''}"
        )

    if not parts:
        parts.append(f"{max(seconds, 1)} seconds")

    return " ".join(parts)


def friendly_error(error):
    if isinstance(error, QuotaExceeded):
        seconds = error.retry_seconds

        if seconds is None:
            return (
                "The AI service could not answer right now. "
                "Please try again in a little while."
            )

        if seconds < 3600:
            return (
                "The request limit has been reached. "
                f"Please try again in about {format_wait(seconds)}."
            )

        reset = datetime.now(timezone.utc) + timedelta(
            seconds=seconds
        )
        local = reset.astimezone(DISPLAY_TZ)

        clock = local.strftime("%I:%M %p").lstrip("0")
        day = f"{local.day} {local.strftime('%b')}"

        return (
            "The request limit has been reached. "
            f"Try again after {clock} IST on {day} "
            f"(about {format_wait(seconds)})."
        )

    message = str(error)

    if len(message) > 500:
        message = message[:500] + "..."

    return f"Something went wrong: {message}"


# ============================================================
# 9. GENERATE TEXT: GROQ FIRST, GEMINI BACKUP
# ============================================================

def generate_text(prompt):
    """
    Attempt Groq first.
    If Groq fails, try Gemini models in sequence.
    """

    groq_error = None

    if groq_client is not None and groq_model is not None:
        try:
            response = groq_client.chat.completions.create(
                model=groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are PaperLens, a research paper "
                            "explainer assistant. Answer using the "
                            "provided research paper content. "
                            "Do not invent facts or use unsupported "
                            "claims about the paper."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=2200,
            )

            answer = response.choices[0].message.content

            if answer and answer.strip():
                print(f"Answered by Groq ({groq_model})")
                return answer.strip()

            raise RuntimeError("Groq returned an empty answer.")

        except Exception as e:
            groq_error = str(e)

            print(
                "Groq failed. Switching to Gemini:",
                groq_error[:300],
            )

    quota_models = []
    unavailable_models = []
    retry_delays = []
    gemini_errors = []

    if client is not None:
        for model in get_model_candidates():
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                answer = getattr(response, "text", None)

                if answer and answer.strip():
                    print(f"Answered by Gemini ({model})")
                    return answer.strip()

                gemini_errors.append(
                    f"{model}: empty response"
                )

            except Exception as e:
                message = str(e)

                print(
                    f"Gemini model {model} failed:",
                    message[:250],
                )

                if (
                    "429" in message
                    or "RESOURCE_EXHAUSTED" in message
                ):
                    quota_models.append(model)

                    delay = parse_retry_seconds(message)

                    if delay is not None:
                        retry_delays.append(delay)

                elif any(
                    code in message
                    for code in (
                        "404",
                        "NOT_FOUND",
                        "503",
                        "UNAVAILABLE",
                    )
                ):
                    unavailable_models.append(model)

                else:
                    gemini_errors.append(
                        f"{model}: {message[:200]}"
                    )

    if quota_models or unavailable_models:
        raise QuotaExceeded(
            quota_models,
            unavailable_models,
            min(retry_delays) if retry_delays else None,
        )

    details = []

    if groq_error:
        details.append(f"Groq error: {groq_error[:250]}")

    details.extend(gemini_errors)

    if client is None:
        details.append("Gemini is not configured.")

    raise RuntimeError(
        "No AI provider could generate an answer. "
        + " | ".join(details)
    )


# ============================================================
# 10. GENERATE A SUMMARY (ONE OR MANY PAPERS)
# ============================================================

def generate_summary(papers):
    per_paper_limit = 60000 // max(len(papers), 1)

    paper_blocks = []

    for paper in papers:
        text = "\n\n".join(
            f"PAGE {page['page']}\n{page['text']}"
            for page in paper["pages"]
        )[:per_paper_limit]

        paper_blocks.append(
            f"===== PAPER: {paper['name']} =====\n{text}"
        )

    all_text = "\n\n".join(paper_blocks)

    if len(papers) == 1:
        prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research paper provided below.
Do not use outside knowledge. Do not invent information.

Write the answer in Markdown using these sections.
Give 1 to 3 short sentences or bullets for each section.

**Research objective**
**Problem statement**
**Methodology**
**Dataset**
**Algorithms and technologies used**
**Main results**
**Limitations**
**Conclusion**

If information is not available in the paper, write:
Not specified in the paper.

Research paper:
{all_text}
"""
    else:
        prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research papers provided below.
Do not use outside knowledge. Do not invent information.

For EACH paper, write a heading with the paper name, followed by
short bullets for: objective, methodology, dataset, main results,
and limitations. Keep each bullet to one sentence.

Then add a final section titled **How these papers relate**
that covers: common ground, key differences, and how one could
build on or complement the other.

If information is not available, write: Not specified in the paper.

Research papers:
{all_text}
"""

    return generate_text(prompt)


# ============================================================
# 11. ASK QUESTIONS ABOUT ONE OR MORE PAPERS
# ============================================================

def ask_question(question, papers, mode):
    query_embedding = embedding_model.encode([question])[0].tolist()

    multi = len(papers) > 1

    per_paper = 5 if not multi else max(2, min(4, 12 // len(papers)))

    chunks = []
    metadata = []
    distances = []
    paper_names = []

    for paper in papers:
        collection = paper["collection"]
        count = collection.count()

        if count == 0:
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(per_paper, count),
            include=["documents", "metadatas", "distances"],
        )

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(doc)
            metadata.append({
                "page": meta.get("page"),
                "paper": paper["name"],
            })
            distances.append(dist)
            paper_names.append(paper["name"])

    if not chunks:
        raise ValueError("No indexed paper content was found.")

    context = "\n\n".join(
        (
            f"SOURCE {i + 1}\n"
            f"PAPER: {metadata[i]['paper']}\n"
            f"PAGE: {metadata[i].get('page', 'Unknown')}\n\n"
            f"TEXT:\n{chunks[i]}"
        )
        for i in range(len(chunks))
    )

    if multi:
        names = ", ".join(paper["name"] for paper in papers)
        scope_rules = f"""
The user has uploaded {len(papers)} papers: {names}.
When the question asks to compare or relate the papers, point out
similarities, differences, and connections between them.
Always say which paper a statement comes from and mention pages
like (Paper name, p. 3).
"""
    else:
        scope_rules = (
            "Mention page numbers like (p. 3) when supported "
            "by the source."
        )

    prompt = f"""
You are PaperLens, a research paper question-answering assistant.

Answer using ONLY the retrieved content from the research paper(s).
Do NOT use outside knowledge. Do NOT invent information.
{scope_rules}

Answer style:
{STYLE_INSTRUCTIONS[mode]}

If the answer cannot be found in the retrieved paper content, say:
The information is not available in the research paper.

User question:
{question}

Retrieved paper content:
{context}
"""

    answer = generate_text(prompt)

    return answer, chunks, metadata, distances


# ============================================================
# 12. HELPERS: CITATIONS AND SHARING
# ============================================================

def format_citations(metadata):
    """Build the 'Source pages' line shown under an answer."""
    by_paper = {}

    for meta in metadata:
        page = meta.get("page")

        if page:
            by_paper.setdefault(meta.get("paper", ""), set()).add(page)

    if not by_paper:
        return ""

    if len(by_paper) == 1:
        pages = sorted(next(iter(by_paper.values())))
        return "Source pages: " + ", ".join(str(p) for p in pages)

    parts = []

    for paper, pages in by_paper.items():
        page_text = ", ".join(str(p) for p in sorted(pages))
        parts.append(f"{html.escape(paper)} (p. {page_text})")

    return "Sources: " + " · ".join(parts)


def build_share_text(chat):
    lines = [
        f"PaperLens - {chat['title']}",
        "Papers: " + ", ".join(p["name"] for p in chat["papers"]),
        "",
    ]

    for item in chat["history"]:
        lines.append(f"Q: {item['question']}")
        lines.append(f"A: {item['answer']}")
        lines.append("")

    return "\n".join(lines).strip()


def short_share_text(text, limit=1200):
    """Share links have URL length limits, so shorten the text."""
    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "...\n(Shortened. Full chat on request.)"


def render_user_bubble(text):
    safe = html.escape(text).replace("\n", "<br>")

    st.markdown(
        '<div class="pl-user-row">'
        f'<div class="pl-user-bubble">{safe}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def section(label):
    st.markdown(
        f'<div class="pl-section">{label}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 13. SIDEBAR
# ============================================================

def chat_button(chat_id, prefix):
    chat = st.session_state.chats[chat_id]

    title = chat["title"]

    if len(title) > 30:
        title = title[:27] + "..."

    is_current = chat_id == st.session_state.current_chat_id

    if st.button(
        title,
        key=f"{prefix}_{chat_id}",
        use_container_width=True,
        type="primary" if is_current else "secondary",
        help=(
            f"{chat['title']} · "
            f"{len(chat['history'])} messages"
        ),
    ):
        open_chat(chat_id)
        st.rerun()


with st.sidebar:

    st.markdown(
        '<div class="pl-logo">PaperLens</div>',
        unsafe_allow_html=True,
    )

    section("Answer style")

    active_mode = st.session_state.explanation_mode or "Simple"

    mode_cols = st.columns(2)

    for col, mode_name in zip(mode_cols, MODES):
        with col:
            st.button(
                mode_name,
                key=f"mode_{mode_name}",
                on_click=set_mode,
                args=(mode_name,),
                use_container_width=True,
                type=(
                    "primary"
                    if mode_name == active_mode
                    else "secondary"
                ),
            )

    st.caption(MODE_HINTS[active_mode])

    st.write("")

    if st.button(
        "New chat",
        key="new_chat_btn",
        icon=":material/add:",
        type="primary",
        use_container_width=True,
    ):
        new_chat()
        st.rerun()

    # Favourites
    pinned_ids = [
        chat_id
        for chat_id in st.session_state.pinned
        if chat_id in st.session_state.chats
    ]

    if pinned_ids:
        section("Favourites")

        with st.container(key="chatlist_fav"):
            for chat_id in pinned_ids:
                chat_button(chat_id, "pinned")

    # Recents
    section("Recents")

    recent_ids = [
        chat_id
        for chat_id in st.session_state.recents
        if (
            chat_id in st.session_state.chats
            and chat_id not in st.session_state.pinned
        )
    ]

    if len(recent_ids) > 4:
        search = st.text_input(
            "Search chats",
            placeholder="Search chats",
            label_visibility="collapsed",
            key="chat_search",
        ).strip().lower()

        if search:
            recent_ids = [
                chat_id
                for chat_id in recent_ids
                if search in st.session_state.chats[chat_id]["title"].lower()
            ]

    if recent_ids:
        with st.container(key="chatlist_recent"):
            for chat_id in recent_ids:
                chat_button(chat_id, "recent")
    else:
        st.caption("No chats yet. Upload a paper to begin.")

    # Chat actions
    current = get_current_chat()

    if current:
        section("Chat actions")

        with st.container(key="chat_actions"):

            st.button(
                "Clear messages",
                key="act_clear",
                icon=":material/ink_eraser:",
                on_click=clear_chat,
                args=(current["id"],),
                use_container_width=True,
            )

            if st.session_state.confirm_delete == current["id"]:
                st.caption("Delete this chat and its messages?")

                yes_col, no_col = st.columns(2)

                with yes_col:
                    if st.button(
                        "Delete",
                        key="act_delete_yes",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.confirm_delete = None
                        delete_chat(current["id"])
                        st.rerun()

                with no_col:
                    st.button(
                        "Cancel",
                        key="act_delete_no",
                        on_click=set_confirm_delete,
                        args=(None,),
                        use_container_width=True,
                    )

            else:
                st.button(
                    "Delete chat",
                    key="act_delete",
                    icon=":material/delete:",
                    on_click=set_confirm_delete,
                    args=(current["id"],),
                    use_container_width=True,
                )


# ============================================================
# 14. MAIN: UPLOAD SCREEN
# ============================================================

current_chat = get_current_chat()

if current_chat is None:

    st.markdown(
        '<div class="pl-hero-title">'
        'Ask questions about any research paper.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="pl-hero-sub">'
        'Upload one or more PDFs and chat with them. '
        'Add several papers to compare and relate them. '
        'Answers include references to source pages.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_upload_{st.session_state.upload_key}",
    )

    if uploaded_files:
        count = len(uploaded_files)

        if st.button(
            f"Start chat with {count} paper{'s' if count != 1 else ''}",
            key="start_chat",
            type="primary",
        ):
            started = False

            with st.spinner("Reading and indexing your papers..."):
                papers, errors = process_uploaded_files(uploaded_files)

            for error in errors:
                st.error(f"Unable to process {error}")

            if papers:
                create_chat(papers)
                started = True

            if started and not errors:
                st.rerun()


# ============================================================
# 15. MAIN: CHAT SCREEN
# ============================================================

else:

    papers = current_chat["papers"]
    history = current_chat["history"]
    multi = len(papers) > 1

    mode = st.session_state.explanation_mode or "Simple"

    # One-time notices (e.g. after adding papers)
    if st.session_state.notice:
        level, message = st.session_state.notice
        st.session_state.notice = None

        if level == "error":
            st.error(message)
        else:
            st.success(message)

    # ---- Header ----
    total_pages = sum(len(p["pages"]) for p in papers)
    total_words = sum(
        len(page["text"].split())
        for paper in papers
        for page in paper["pages"]
    )

    paper_label = (
        f"{len(papers)} papers" if multi else "1 paper"
    )

    st.markdown(
        (
            '<p class="pl-title">'
            f'{html.escape(current_chat["title"])}'
            '</p>'
            '<p class="pl-meta">'
            f'{paper_label} · {total_pages} pages · '
            f'{total_words:,} words · {mode} answers'
            '</p>'
        ),
        unsafe_allow_html=True,
    )

    if multi:
        st.caption(
            "Papers in this chat: "
            + " · ".join(p["name"] for p in papers)
        )

    # ---- Action row: Summarize, Favourite, Share ----
    is_fav = current_chat["id"] in st.session_state.pinned

    _, sum_col, fav_col, share_col = st.columns([2.2, 1.6, 1.4, 1.1])

    with sum_col:
        st.button(
            "Summarize papers" if multi else "Summarize paper",
            key="top_summary",
            on_click=ask_summary,
            args=(multi,),
            use_container_width=True,
        )

    with fav_col:
        st.button(
            "Favourited" if is_fav else "Favourite",
            key="top_fav",
            icon=":material/star:",
            on_click=toggle_pin,
            args=(current_chat["id"],),
            type="primary" if is_fav else "secondary",
            use_container_width=True,
        )

    with share_col:
        with st.popover(
            "Share",
            icon=":material/share:",
            use_container_width=True,
        ):
            if not history:
                st.caption(
                    "Ask a question first, then share the conversation."
                )
            else:
                full_text = build_share_text(current_chat)
                short_text = short_share_text(full_text)
                encoded = quote(short_text)
                subject = quote(f"PaperLens: {current_chat['title']}")

                st.link_button(
                    "WhatsApp",
                    f"https://wa.me/?text={encoded}",
                    use_container_width=True,
                )

                st.link_button(
                    "Telegram",
                    f"https://t.me/share/url?url=%20&text={encoded}",
                    use_container_width=True,
                )

                st.link_button(
                    "Email",
                    f"mailto:?subject={subject}&body={encoded}",
                    use_container_width=True,
                )

                safe_name = re.sub(
                    r"[^A-Za-z0-9_-]+", "_", current_chat["title"]
                ).strip("_") or "paperlens_chat"

                st.download_button(
                    "Download as text file",
                    data=full_text,
                    file_name=f"{safe_name}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                st.caption("Copy the full conversation:")
                st.code(full_text, language=None)

    # ---- Add more papers ----
    with st.expander("Add more papers to compare"):
        extra_files = st.file_uploader(
            "Add PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"add_upload_{st.session_state.add_key}",
            label_visibility="collapsed",
        )

        if extra_files:
            if st.button(
                f"Add {len(extra_files)} paper"
                f"{'s' if len(extra_files) != 1 else ''} to this chat",
                key="add_papers_btn",
                type="primary",
            ):
                with st.spinner("Reading and indexing..."):
                    new_papers, errors = process_uploaded_files(
                        extra_files
                    )

                added = add_papers_to_chat(current_chat, new_papers)

                st.session_state.add_key += 1

                if errors:
                    st.session_state.notice = (
                        "error",
                        "Could not add: " + "; ".join(errors),
                    )
                else:
                    st.session_state.notice = (
                        "success",
                        f"Added {added} paper"
                        f"{'s' if added != 1 else ''}. "
                        "You can now ask questions across all papers.",
                    )

                st.rerun()

    # ---- Empty state and suggestions ----
    pending = st.session_state.pending

    if not history and not pending:

        st.markdown(
            '<div class="pl-empty">'
            '<b>What would you like to know?</b>'
            '<span>Ask a question below, or try one of these.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if multi:
            suggestions = [
                "How are these papers related?",
                "Compare the methods used in these papers.",
                "What are the key differences in their results?",
            ]
        else:
            suggestions = [
                "What problem does this paper solve?",
                "Explain the method step by step.",
                "What are the main results and limitations?",
            ]

        for i, suggestion in enumerate(suggestions):
            st.button(
                suggestion,
                key=f"suggest_{i}",
                on_click=ask_suggestion,
                args=(suggestion,),
                use_container_width=True,
            )

    # ---- Chat history: questions right, answers left ----
    for item in history:

        render_user_bubble(item["question"])

        with st.container(border=True):
            st.markdown(item["answer"])

            metadata = item.get("metadata", [])
            chunks = item.get("chunks", [])
            distances = item.get("distances", [])

            citation_line = format_citations(metadata)

            if citation_line:
                st.markdown(
                    f'<div class="pl-cites">{citation_line}</div>',
                    unsafe_allow_html=True,
                )

            if chunks:
                with st.expander(
                    f"Sources ({len(chunks)} passages)"
                ):
                    for i, chunk in enumerate(chunks):
                        meta = metadata[i] if i < len(metadata) else {}
                        page_number = meta.get("page", "?")
                        paper_name = meta.get("paper", "")

                        distance = (
                            distances[i]
                            if i < len(distances)
                            else None
                        )

                        label = f"**Page {page_number}**"

                        if multi and paper_name:
                            label = f"**{paper_name}** · page {page_number}"

                        if distance is not None:
                            label += f" · distance {distance:.3f}"

                        st.markdown(label)
                        st.caption(chunk)

    # ---- Process pending question or summary ----
    if pending:

        render_user_bubble(pending["question"])

        failed = False
        result = None

        with st.container(border=True):
            with st.spinner(
                "Reading the papers and generating your answer..."
            ):
                try:
                    if pending["kind"] == "summary":
                        result = (
                            generate_summary(papers),
                            [],
                            [],
                            [],
                        )
                    else:
                        result = ask_question(
                            pending["question"],
                            papers,
                            mode,
                        )

                except Exception as e:
                    failed = True

                    if isinstance(e, QuotaExceeded):
                        st.warning(friendly_error(e))
                    else:
                        st.error(friendly_error(e))

        if failed:
            st.session_state.pending = None

        elif result is not None:
            answer, chunks, metadata, distances = result

            history.append({
                "question": pending["question"],
                "answer": answer,
                "chunks": chunks,
                "metadata": metadata,
                "distances": distances,
            })

            move_to_top(current_chat["id"])

            st.session_state.pending = None

            st.rerun()

    # ---- Chat input (with multi-PDF attach when supported) ----
    try:
        submitted = st.chat_input(
            "Ask about your papers, or attach PDFs to add them...",
            accept_file="multiple",
            file_type=["pdf"],
        )
    except TypeError:
        # Older Streamlit versions: text only.
        submitted = st.chat_input("Ask about your papers...")

    if submitted:

        if isinstance(submitted, str):
            text = submitted
            attached = []
        else:
            text = getattr(submitted, "text", "") or ""
            attached = list(getattr(submitted, "files", []) or [])

        if attached:
            with st.spinner("Reading and indexing attached papers..."):
                new_papers, errors = process_uploaded_files(attached)

            added = add_papers_to_chat(current_chat, new_papers)

            if errors:
                st.session_state.notice = (
                    "error",
                    "Could not add: " + "; ".join(errors),
                )
            elif added:
                st.session_state.notice = (
                    "success",
                    f"Added {added} paper{'s' if added != 1 else ''} "
                    "to this chat.",
                )

        if text.strip():
            st.session_state.pending = {
                "kind": "question",
                "question": text.strip(),
            }

        st.rerun()
