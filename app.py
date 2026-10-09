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
RED = "#C0392B"


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

/* ---- Hide Streamlit's menu, deploy button and footer ----
   The toolbar itself stays, because the "open sidebar" arrow
   can live inside it in newer Streamlit versions. */
footer,
#MainMenu,
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stToolbarActions"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
.stDeployButton {{
    display: none !important;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

/* ---- Always keep the sidebar open/close arrows visible ---- */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}}

[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    z-index: 999999;
}}

.block-container {{
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}}

/* ---- Typography ---- */
.pl-logo {{
    font: 600 1.7rem 'Newsreader', Georgia, serif;
    margin: 0;
    color: {BLUE};
    line-height: 1.15;
}}

.pl-section {{
    font-size: .8rem;
    font-weight: 500;
    color: #7A8899;
    margin: .7rem 0 0 0;
    padding-left: .1rem;
}}

.pl-section.tight {{
    margin-top: -.2rem;
}}

.pl-title {{
    font: 600 1.55rem 'Newsreader', Georgia, serif;
    line-height: 1.25;
    margin: 0 0 .15rem 0;
}}

.pl-meta {{
    color: #66758a;
    font-size: .9rem;
    margin: 0;
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

/* ---- Sidebar layout and spacing ---- */
[data-testid="stSidebar"] {{
    background: #F7F9FC;
}}

[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
    padding: .75rem 1rem .25rem 1rem;
    height: auto;
    min-height: 0;
}}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    padding: 0 1rem 1.5rem 1rem;
}}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: .5rem;
}}

[data-testid="stSidebar"] button {{
    border-radius: 8px;
    min-height: 2.4rem;
    padding-top: .3rem;
    padding-bottom: .3rem;
}}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    margin-top: -.15rem;
    padding-left: .1rem;
    line-height: 1.35;
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

/* ---- Answer style: joined Simple | Technical switch ---- */
.st-key-mode_switch [data-testid="stHorizontalBlock"] {{
    gap: 0 !important;
}}

.st-key-mode_switch [data-testid="stColumn"] {{
    min-width: 0 !important;
}}

.st-key-mode_switch [data-testid="stColumn"]:first-child button {{
    border-radius: 8px 0 0 8px !important;
}}

.st-key-mode_switch [data-testid="stColumn"]:last-child button {{
    border-radius: 0 8px 8px 0 !important;
    margin-left: -1px;
}}

.st-key-mode_switch button[kind="secondary"],
.st-key-mode_switch [data-testid="stBaseButton-secondary"] {{
    background: #FFFFFF !important;
    border: 1px solid #C9D6E4 !important;
    color: #33475B !important;
}}

.st-key-mode_switch button[kind="primary"],
.st-key-mode_switch [data-testid="stBaseButton-primary"] {{
    background: {BLUE} !important;
    border: 1px solid {BLUE} !important;
    color: #FFFFFF !important;
    font-weight: 600;
    position: relative;
    z-index: 1;
}}

/* ---- Chat lists (Favourites and Recents) ---- */
[class*="st-key-chatlist"] [data-testid="stVerticalBlock"] {{
    gap: .2rem !important;
}}

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
.st-key-chat_actions [data-testid="stVerticalBlock"] {{
    gap: .2rem !important;
}}

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
    border-color: transparent !important;
    color: {BLUE} !important;
}}

.st-key-act_delete button {{
    color: {RED} !important;
}}

.st-key-act_delete button:hover {{
    background: #FBECEA !important;
    color: {RED} !important;
}}

/* Delete / Cancel confirm buttons stay bordered */
.st-key-chat_actions [data-testid="stHorizontalBlock"] button {{
    justify-content: center !important;
    border: 1px solid #D5DEE9 !important;
}}

.st-key-chat_actions [data-testid="stHorizontalBlock"] button[kind="primary"],
.st-key-chat_actions [data-testid="stHorizontalBlock"] [data-testid="stBaseButton-primary"] {{
    background: {BLUE} !important;
    border-color: {BLUE} !important;
    color: #FFFFFF !important;
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
    seen = set()

    for uploaded in files:
        try:
            paper = process_pdf(uploaded.getvalue(), uploaded.name)

            # Skip the same PDF if it was selected twice
            if paper["id"] in seen:
                continue

            seen.add(paper["id"])
            papers.append(paper)

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
    """Extract a retry
