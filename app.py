
import html
import re
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from io import BytesIO

import streamlit as st
import chromadb
from groq import Groq
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_MODEL = "gemini-3-flash-preview"

FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# Groq models are attempted in this order.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
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


# ============================================================
# STYLING
# UI DESIGN PRESERVED
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Public Sans', system-ui, sans-serif;
}

footer, #MainMenu {
    display: none;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Keep the sidebar open/close controls visible */
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
}

/* Centered reading column for chat */
.block-container {
    max-width: 880px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* Sidebar */
.pl-logo-name {
    font: 600 1.45rem 'Newsreader', Georgia, serif;
    line-height: 1.1;
    margin: 0 0 .3rem 0;
}

/* Recent chats */
[data-testid="stSidebar"] [class*="st-key-recent_"] button,
[data-testid="stSidebar"] [class*="st-key-pinned_"] button {
    background: transparent;
    border: none;
    border-radius: 8px;
    justify-content: flex-start;
    text-align: left;
    font-weight: 500;
    padding: .5rem .75rem;
    min-height: 0;
    box-shadow: none;
}

[data-testid="stSidebar"] [class*="st-key-recent_"] button:hover,
[data-testid="stSidebar"] [class*="st-key-pinned_"] button:hover {
    background: rgba(47,93,138,.08);
}

[data-testid="stSidebar"] [class*="st-key-recent_"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] [class*="st-key-pinned_"] [data-testid="stBaseButton-primary"] {
    background: #E6EEF7;
    color: #1B2430;
    box-shadow: inset 3px 0 0 #2F5D8A;
}

.st-key-new_chat_btn button {
    background: #2F5D8A !important;
    color: #fff !important;
    border: none !important;
    justify-content: center !important;
    font-weight: 600;
    border-radius: 9px;
}

.st-key-new_chat_btn button:hover {
    background: #254B72 !important;
}

/* Sidebar spacing */
[data-testid="stSidebarHeader"] {
    height: auto;
    padding: .6rem 1rem 0 1rem;
}

[data-testid="stSidebarUserContent"] {
    padding-top: .3rem;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: .6rem;
}

/* Left-align text inside sidebar buttons */
[data-testid="stSidebar"] .stButton button > div {
    justify-content: flex-start;
    width: 100%;
}

[data-testid="stSidebar"] .stButton button p {
    text-align: left;
}

.st-key-new_chat_btn button > div {
    justify-content: center !important;
}

/* Chat actions */
[class*="st-key-act_"] button {
    background: transparent;
    border: none;
    box-shadow: none;
    border-radius: 8px;
    padding: .4rem .75rem;
    min-height: 0;
    font-weight: 500;
    justify-content: flex-start;
}

[class*="st-key-act_"] button:hover {
    background: rgba(47,93,138,.08);
}

.st-key-act_delete button,
.st-key-act_delete_yes button {
    color: #B3382C;
}

.st-key-act_delete button:hover {
    background: rgba(179,56,44,.08);
}

.st-key-act_delete_yes button,
.st-key-act_delete_no button {
    justify-content: center !important;
    border: 1px solid #D5DDE6 !important;
}

.st-key-act_delete_yes button > div,
.st-key-act_delete_no button > div {
    justify-content: center !important;
}

/* Answer style buttons */
[class*="st-key-mode_"] [data-testid="stBaseButton-primary"] {
    background: #2F5D8A !important;
    border: 1px solid #2F5D8A !important;
    color: #fff !important;
}

[class*="st-key-mode_"] [data-testid="stBaseButton-primary"] p {
    color: #fff !important;
}

[class*="st-key-mode_"] [data-testid="stBaseButton-secondary"] {
    background: #fff !important;
    border: 1px solid #D5DDE6 !important;
    color: #1B2430 !important;
}

[class*="st-key-mode_"] [data-testid="stBaseButton-secondary"]:hover {
    border-color: #2F5D8A !important;
    color: #2F5D8A !important;
}

[class*="st-key-mode_"] button > div {
    justify-content: center !important;
}

[class*="st-key-mode_"] button p {
    text-align: center !important;
}

/* Blue accent */
[data-testid="stBaseButton-segmented_controlActive"],
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: #2F5D8A !important;
    border-color: #2F5D8A !important;
    color: #fff !important;
}

[data-testid="stBaseButton-segmented_controlActive"] p,
[data-testid="stBaseButton-segmented_controlActive"] span,
[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
[data-testid="stSegmentedControl"] button[aria-pressed="true"] p {
    color: #fff !important;
}

[data-testid="stBaseButton-segmented_control"]:hover {
    border-color: #2F5D8A;
    color: #2F5D8A;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #2F5D8A !important;
    box-shadow: 0 0 0 1px #2F5D8A;
}

[data-testid="stTextInput"] input:focus {
    border-color: #2F5D8A !important;
    box-shadow: 0 0 0 1px #2F5D8A !important;
}

/* Header */
.pl-title {
    font: 600 1.55rem 'Newsreader', Georgia, serif;
    margin: 0;
    line-height: 1.25;
}

.pl-meta {
    color: #66758a;
    font-size: .85rem;
    margin: .2rem 0 .8rem 0;
}

/* Chat */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none;
}

[data-testid="stChatMessage"] {
    width: fit-content;
    max-width: 88%;
    border-radius: 16px;
    padding: .8rem 1.1rem;
    background: #F4F7FA;
    border: 1px solid #E3E9F0;
    margin-right: auto;
}

/* User messages */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto;
    margin-right: 0;
    max-width: 72%;
    background: #E1ECF7;
    border-color: #C9DCEF;
    border-bottom-right-radius: 4px;
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-bottom-left-radius: 4px;
    background: #FFFFFF;
    border-color: #DDE5EE;
}

.pl-cites {
    color: #66758a;
    font-size: .82rem;
    margin-top: .4rem;
}

.pl-empty {
    text-align: center;
    padding: 3rem 1rem 1rem 1rem;
}

.pl-empty b {
    font: 600 1.5rem 'Newsreader', Georgia, serif;
}

.pl-empty span {
    display: block;
    color: #66758a;
    margin-top: .3rem;
}

/* Upload screen */
.pl-hero-title {
    font: 600 2.8rem 'Newsreader', Georgia, serif;
    line-height: 1.12;
    margin: 2rem 0 .5rem 0;
}

.pl-hero-sub {
    font-size: 1.05rem;
    color: #66758a;
    margin-bottom: 1.4rem;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #9DB6CF;
    border-radius: 16px;
    padding: 2rem 1.2rem;
    background: #F7FAFD;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "current_chat_id": None,
    "chats": {},
    "recents": [],
    "pinned": [],
    "explanation_mode": "Simple",
    "upload_key": 0,
    "pending": None,
    "confirm_delete": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# API KEYS AND AI CLIENTS
# GROQ FIRST, GEMINI FALLBACK
# ============================================================

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

groq_client = None
gemini_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.warning(f"Unable to initialize Groq: {e}")

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.warning(f"Unable to initialize Gemini: {e}")

if groq_client is None and gemini_client is None:
    st.error(
        "No AI provider is configured. Add GROQ_API_KEY or "
        "GEMINI_API_KEY to your Streamlit Secrets."
    )
    st.stop()


# ============================================================
# EMBEDDING AND VECTOR DATABASE
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
# PDF PROCESSING
# ============================================================

def extract_pdf_pages(file_bytes):
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
    digest = hashlib.md5(file_bytes).hexdigest()[:10]
    return f"research_paper_{digest}"


def process_pdf(file_bytes):
    pages = extract_pdf_pages(file_bytes)

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF. "
            "The PDF may contain scanned images instead of selectable text."
        )

    chunks, page_numbers = create_chunks(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False,
    )

    collection = chroma_client.get_or_create_collection(
        name=create_collection_name(file_bytes)
    )

    if collection.count() == 0:
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[
                {"page": page}
                for page in page_numbers
            ],
        )

    return pages, collection


# ============================================================
# CHAT MANAGEMENT
# ============================================================

def create_chat(paper_name, collection, pages):
    chat_id = uuid.uuid4().hex[:10]

    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "title": paper_name,
        "paper_name": paper_name,
        "collection": collection,
        "pages": pages,
        "history": [],
    }

    st.session_state.recents.insert(0, chat_id)
    st.session_state.current_chat_id = chat_id
    st.session_state.pending = None

    return chat_id


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


def ask_summary():
    st.session_state.pending = {
        "kind": "summary",
        "question": "Summarize this paper.",
    }


# ============================================================
# AI FUNCTIONS
# GROQ -> GEMINI -> GEMINI FALLBACK MODELS
# ============================================================

class QuotaExceeded(Exception):
    """Raised when configured models fail due to quota limits."""

    def __init__(
        self,
        quota_models,
        unavailable_models,
        retry_seconds=None,
    ):
        super().__init__("AI model quota exceeded")
        self.quota_models = quota_models
        self.unavailable_models = unavailable_models
        self.retry_seconds = retry_seconds


def parse_retry_seconds(message):
    """Extract retry duration from provider error messages."""

    match = re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
        message,
        re.IGNORECASE,
    )

    if match:
        return float(match.group(1))

    match = re.search(
        r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?",
        message,
        re.IGNORECASE,
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
    primary = get_secret("GEMINI_MODEL", PRIMARY_MODEL)

    return [primary] + [
        model for model in FALLBACK_MODELS
        if model != primary
    ]


def get_groq_model_candidates():
    primary = get_secret(
        "GROQ_MODEL",
        GROQ_MODELS[0],
    )

    return [primary] + [
        model for model in GROQ_MODELS
        if model != primary
    ]


def is_quota_error(message):
    lower_message = message.lower()

    return (
        "429" in message
        or "resource_exhausted" in lower_message
        or "rate_limit_exceeded" in lower_message
        or "rate limit" in lower_message
        or "quota" in lower_message
    )


def is_unavailable_error(message):
    lower_message = message.lower()

    return (
        any(code in message for code in ("404", "503"))
        or "unavailable" in lower_message
        or "model_not_found" in lower_message
        or "model not found" in lower_message
        or "decommissioned" in lower_message
    )


def generate_text(prompt):
    """
    First try Groq.
    If Groq models fail, try Gemini.
    If the first Gemini model fails, try the remaining Gemini models.
    """

    quota_models = []
    unavailable_models = []
    retry_delays = []
    other_errors = []

    # --------------------------------------------------------
    # 1. GROQ MODELS
    # --------------------------------------------------------

    if groq_client is not None:
        for model in get_groq_model_candidates():
            try:
                response = groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are PaperLens, a research paper "
                                "analysis assistant. Follow the prompt "
                                "carefully and do not invent facts."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.2,
                )

                answer = response.choices[0].message.content

                if answer and answer.strip():
                    return answer.strip()

                other_errors.append(
                    f"Groq model {model} returned an empty answer."
                )

            except Exception as e:
                message = str(e)

                if is_quota_error(message):
                    quota_models.append(f"Groq: {model}")

                    delay = parse_retry_seconds(message)

                    if delay is not None:
                        retry_delays.append(delay)

                elif is_unavailable_error(message):
                    unavailable_models.append(f"Groq: {model}")

                else:
                    other_errors.append(
                        f"Groq {model}: {message[:200]}"
                    )

    # --------------------------------------------------------
    # 2. GEMINI MODELS
    # --------------------------------------------------------

    if gemini_client is not None:
        for model in get_model_candidates():
            try:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                answer = response.text

                if answer and answer.strip():
                    return answer.strip()

                other_errors.append(
                    f"Gemini model {model} returned an empty answer."
                )

            except Exception as e:
                message = str(e)

                if is_quota_error(message):
                    quota_models.append(f"Gemini: {model}")

                    delay = parse_retry_seconds(message)

                    if delay is not None:
                        retry_delays.append(delay)

                elif is_unavailable_error(message):
                    unavailable_models.append(f"Gemini: {model}")

                else:
                    other_errors.append(
                        f"Gemini {model}: {message[:200]}"
                    )

    # --------------------------------------------------------
    # 3. ALL PROVIDERS FAILED
    # --------------------------------------------------------

    # Report a quota error when every failure was a quota issue
    # or a model-availability issue.
    if quota_models and not other_errors:
        raise QuotaExceeded(
            quota_models,
            unavailable_models,
            min(retry_delays) if retry_delays else None,
        )

    if unavailable_models and not other_errors and not quota_models:
        raise RuntimeError(
            "All configured AI models are unavailable. "
            "Check your model names and API provider settings."
        )

    details = []

    if quota_models:
        details.append(
            "Quota/rate-limit failures: "
            + ", ".join(quota_models)
        )

    if unavailable_models:
        details.append(
            "Unavailable models: "
            + ", ".join(unavailable_models)
        )

    if other_errors:
        details.append(
            "Other errors:\n" + "\n".join(other_errors[:4])
        )

    raise RuntimeError(
        "Groq and Gemini could not generate an answer.\n"
        + "\n".join(details)
    )


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
                "The configured AI models have reached their "
                "request limits or are unavailable. Please try again "
                "later, or check your Groq and Gemini quotas."
            )

        if seconds < 3600:
            return (
                "The AI request limit has been reached. "
                f"Please try again in about {format_wait(seconds)}."
            )

        reset = datetime.now(timezone.utc) + timedelta(
            seconds=seconds
        )

        local = reset.astimezone(DISPLAY_TZ)
        clock = local.strftime("%I:%M %p").lstrip("0")
        day = f"{local.day} {local.strftime('%b')}"

        return (
            "The AI request limit has been reached. "
            f"Please try again after {clock} IST on {day} "
            f"(in about {format_wait(seconds)})."
        )

    message = str(error)

    if len(message) > 500:
        message = message[:500] + "..."

    return (
        "Something went wrong while contacting the AI providers: "
        + message
    )


# ============================================================
# PAPER SUMMARY
# ============================================================

def generate_summary(pages):
    paper_text = "\n\n".join(
        f"PAGE {page['page']}\n{page['text']}"
        for page in pages
    )[:60000]

    prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research paper provided below.
Do not use outside knowledge. Do not invent information.

Write the answer in Markdown using EXACTLY these sections,
each as a bold heading followed by 1 to 3 short sentences or bullets:

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
{paper_text}
"""

    return generate_text(prompt)


# ============================================================
# QUESTION ANSWERING WITH RETRIEVAL
# ============================================================

def ask_question(question, collection, mode):
    query_embedding = embedding_model.encode([question])[0]

    count = collection.count()

    if count == 0:
        raise ValueError(
            "No indexed paper content was found. Please upload the PDF again."
        )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, count),
    )

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    context = "\n\n".join(
        (
            f"SOURCE {i + 1}\n"
            f"PAGE: {metadata[i].get('page', 'Unknown')}\n\n"
            f"TEXT:\n{chunks[i]}"
        )
        for i in range(len(chunks))
    )

    prompt = f"""
You are PaperLens, a research paper question-answering assistant.

Answer using ONLY the retrieved content from the research paper.
Do NOT use outside knowledge. Do NOT invent information.
Mention page numbers like (p. 3) when you use a source.

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
# SIDEBAR
# ============================================================

def chat_button(chat_id, prefix):
    chat = st.session_state.chats[chat_id]
    title = chat["title"]

    if len(title) > 32:
        title = title[:29] + "..."

    is_current = chat_id == st.session_state.current_chat_id

    if st.button(
        title,
        key=f"{prefix}_{chat_id}",
        use_container_width=True,
        type="primary" if is_current else "secondary",
        help=(
            f"{chat['paper_name']} · "
            f"{len(chat['history'])} messages"
        ),
    ):
        open_chat(chat_id)
        st.rerun()


with st.sidebar:
    st.markdown(
        '<div class="pl-logo-name">PaperLens</div>',
        unsafe_allow_html=True,
    )

    # Answer style
    st.caption("Answer style")

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

    if st.button(
        "New chat",
        key="new_chat_btn",
        use_container_width=True,
    ):
        new_chat()
        st.rerun()

    # Pinned chats
    pinned_ids = [
        chat_id
        for chat_id in st.session_state.pinned
        if chat_id in st.session_state.chats
    ]

    if pinned_ids:
        st.caption("Pinned")

        for chat_id in pinned_ids:
            chat_button(chat_id, "pinned")

    # Recent chats
    st.caption("Recents")

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
                if (
                    search
                    in st.session_state.chats[chat_id]["title"].lower()
                    or search
                    in st.session_state.chats[chat_id]["paper_name"].lower()
                )
            ]

    if recent_ids:
        for chat_id in recent_ids:
            chat_button(chat_id, "recent")
    else:
        st.caption("No chats yet. Upload a paper to begin.")

    # Actions for the open chat
    current = get_current_chat()

    if current:
        st.caption("Chat actions")

        is_pinned = current["id"] in st.session_state.pinned

        st.button(
            "Unpin chat" if is_pinned else "Pin chat",
            key="act_pin",
            icon=":material/push_pin:",
            on_click=toggle_pin,
            args=(current["id"],),
            use_container_width=True,
        )

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
# MAIN: UPLOAD SCREEN
# ============================================================

current_chat = get_current_chat()

if current_chat is None:
    st.markdown(
        '<div class="pl-hero-title">'
        "Ask questions about any research paper."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="pl-hero-sub">'
        "Upload a PDF and chat with it. "
        "Every answer points to the pages it came from."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload research paper",
        type=["pdf"],
        key=f"pdf_upload_{st.session_state.upload_key}",
        label_visibility="collapsed",
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()

        # Prevent reprocessing the same uploaded file on reruns.
        upload_signature = hashlib.md5(file_bytes).hexdigest()

        if (
            st.session_state.get("processed_upload_signature")
            != upload_signature
        ):
            with st.spinner("Reading and indexing the paper..."):
                try:
                    pages, collection = process_pdf(file_bytes)

                    create_chat(
                        uploaded_file.name,
                        collection,
                        pages,
                    )

                    st.session_state.processed_upload_signature = (
                        upload_signature
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"Unable to process PDF: {e}")


# ============================================================
# MAIN: CHAT
# ============================================================

else:
    pages = current_chat["pages"]
    collection = current_chat["collection"]
    history = current_chat["history"]

    mode = st.session_state.explanation_mode or "Simple"

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    total_words = sum(
        len(page["text"].split())
        for page in pages
    )

    title_col, summary_col = st.columns([4, 1])

    with title_col:
        st.markdown(
            f'<p class="pl-title">'
            f'{html.escape(current_chat["paper_name"])}</p>'
            f'<p class="pl-meta">'
            f'{len(pages)} pages · {total_words:,} words · '
            f'{mode} answers</p>',
            unsafe_allow_html=True,
        )

    with summary_col:
        st.button(
            "Summarize paper",
            key="top_summary",
            on_click=ask_summary,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------------

    pending = st.session_state.pending

    if not history and not pending:
        st.markdown(
            '<div class="pl-empty">'
            '<b>What would you like to know?</b>'
            '<span>Ask a question below, or try one of these.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

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

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for item in history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

            metadata = item.get("metadata", [])
            chunks = item.get("chunks", [])
            distances = item.get("distances", [])

            cited_pages = sorted({
                metadata_item.get("page")
                for metadata_item in metadata
                if metadata_item.get("page")
            })

            if cited_pages:
                st.markdown(
                    '<div class="pl-cites">Pages: '
                    + ", ".join(str(page) for page in cited_pages)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            if chunks:
                with st.expander(
                    f"Sources ({len(chunks)} passages)"
                ):
                    for i, chunk in enumerate(chunks):
                        page_number = metadata[i].get("page", "?")

                        distance_text = ""

                        if i < len(distances):
                            distance_text = (
                                f" · distance {distances[i]:.3f}"
                            )

                        st.markdown(
                            f"**Page {page_number}**{distance_text}"
                        )

                        st.caption(chunk)

    # --------------------------------------------------------
    # PROCESS PENDING REQUEST
    # --------------------------------------------------------

    if pending:
        with st.chat_message("user"):
            st.write(pending["question"])

        with st.chat_message("assistant"):
            with st.spinner("Reading the paper..."):
                try:
                    if pending["kind"] == "summary":
                        answer = generate_summary(pages)

                        chunks = []
                        metadata = []
                        distances = []

                    else:
                        (
                            answer,
                            chunks,
                            metadata,
                            distances,
                        ) = ask_question(
                            pending["question"],
                            collection,
                            mode,
                        )

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

                except Exception as e:
                    st.session_state.pending = None

                    if isinstance(e, QuotaExceeded):
                        st.warning(friendly_error(e))
                    else:
                        st.error(friendly_error(e))

    # --------------------------------------------------------
    # CHAT INPUT
    # Keep Streamlit's normal send arrow
    # --------------------------------------------------------

    question = st.chat_input("Ask about this paper...")

    if question and question.strip():
        st.session_state.pending = {
            "kind": "question",
            "question": question.strip(),
        }

        st.rerun()
