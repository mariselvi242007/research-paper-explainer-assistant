
import html
import re
import uuid
import hashlib
from io import BytesIO
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


# ============================================================
# 2. STYLING
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@400;600&family=Public+Sans:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Public Sans', system-ui, sans-serif;
}

footer, #MainMenu {
    display: none;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

.pl-logo {
    font: 600 1.7rem 'Newsreader', Georgia, serif;
    margin-bottom: .8rem;
}

.pl-title {
    font: 600 1.65rem 'Newsreader', Georgia, serif;
    line-height: 1.3;
}

.pl-meta {
    color: #66758a;
    font-size: .85rem;
}

.pl-hero-title {
    font: 600 2.8rem 'Newsreader', Georgia, serif;
    line-height: 1.15;
    margin: 2rem 0 .6rem 0;
}

.pl-hero-sub {
    color: #66758a;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

.pl-empty {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
}

.pl-empty b {
    font: 600 1.5rem 'Newsreader', Georgia, serif;
}

.pl-empty span {
    display: block;
    color: #66758a;
    margin-top: .4rem;
}

.pl-cites {
    color: #66758a;
    font-size: .82rem;
    margin-top: .5rem;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
    border: 1px solid #E3E9F0;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #2F5D8A !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #9DB6CF;
    border-radius: 16px;
    padding: 1.5rem;
    background: #F7FAFD;
}

[data-testid="stSidebar"] {
    background: #F7F9FC;
}

[data-testid="stSidebar"] button {
    border-radius: 8px;
}

.stButton button {
    transition: all .15s ease;
}

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
    "pending": None,
    "confirm_delete": None,
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

        # Check the models available to this API key.
        available_models = groq_client.models.list()
        available_ids = [
            model.id for model in available_models.data
        ]

        for candidate in PREFERRED_GROQ_MODELS:
            if candidate in available_ids:
                groq_model = candidate
                break

        # If none of the preferred IDs is available, try other
        # compatible model families listed by the account.
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


# ============================================================
# 5. MODEL STATUS
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="pl-logo">📚 PaperLens</div>',
        unsafe_allow_html=True,
    )

    if groq_client is not None and groq_model:
        st.success("Primary AI: Groq")
        st.caption(f"Groq model: {groq_model}")
    else:
        st.warning("Groq is unavailable. Gemini will be tried first.")

    if client is not None:
        st.caption("Backup AI: Gemini")
        st.caption(f"Gemini model: {PRIMARY_MODEL}")
    else:
        st.warning("Gemini backup is not configured.")


if not (
    groq_client is not None and groq_model is not None
) and client is None:
    st.error(
        "No AI provider is available. Add GROQ_API_KEY and/or "
        "GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()


# ============================================================
# 6. EMBEDDING MODEL AND CHROMADB
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
# 7. PDF PROCESSING
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


def process_pdf(file_bytes):
    """Extract, embed, and index the uploaded PDF."""
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

    # Index the PDF only if its collection is empty.
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

    return pages, collection


# ============================================================
# 8. CHAT MANAGEMENT
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
# 9. AI ERROR HANDLING
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
                "The available Gemini models could not answer. "
                "Check your quota, model access, and API settings."
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
# 10. GENERATE TEXT: GROQ FIRST, GEMINI BACKUP
# ============================================================

def generate_text(prompt):
    """
    Attempt Groq first.
    If Groq fails, try Gemini models in sequence.
    """

    groq_error = None

    # --------------------------------------------------------
    # FIRST: GROQ
    # --------------------------------------------------------
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
                max_tokens=1800,
            )

            answer = response.choices[0].message.content

            if answer and answer.strip():
                return (
                    answer.strip()
                    + f"\n\n*AI model: Groq ({groq_model})*"
                )

            raise RuntimeError("Groq returned an empty answer.")

        except Exception as e:
            groq_error = str(e)

            print(
                "Groq failed. Switching to Gemini:",
                groq_error[:300],
            )

    # --------------------------------------------------------
    # SECOND: GEMINI
    # --------------------------------------------------------
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
                    return (
                        answer.strip()
                        + f"\n\n*AI model: Gemini ({model})*"
                    )

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

    # --------------------------------------------------------
    # BOTH PROVIDERS FAILED
    # --------------------------------------------------------
    if quota_models or unavailable_models:
        raise QuotaExceeded(
            quota_models,
            unavailable_models,
            min(retry_delays) if retry_delays else None,
        )

    details = []

    if groq_error:
        details.append(
            f"Groq error: {groq_error[:250]}"
        )

    details.extend(gemini_errors)

    if client is None:
        details.append("Gemini is not configured.")

    raise RuntimeError(
        "No AI provider could generate an answer. "
        + " | ".join(details)
    )


# ============================================================
# 11. GENERATE A PAPER SUMMARY
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
{paper_text}
"""

    return generate_text(prompt)


# ============================================================
# 12. ASK QUESTIONS ABOUT THE PAPER
# ============================================================

def ask_question(question, collection, mode):
    query_embedding = embedding_model.encode([question])[0]

    number_of_chunks = collection.count()

    if number_of_chunks == 0:
        raise ValueError("No indexed paper content was found.")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, number_of_chunks),
        include=["documents", "metadatas", "distances"],
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
Mention page numbers like (p. 3) when supported by the source.

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
# 13. SIDEBAR CHAT BUTTON
# ============================================================

def chat_button(chat_id, prefix):
    chat = st.session_state.chats[chat_id]

    title = chat["title"]

    if len(title) > 32:
        title = title[:29] + "..."

    is_current = (
        chat_id == st.session_state.current_chat_id
    )

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


# ============================================================
# 14. SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.caption("Answer style")

    active_mode = (
        st.session_state.explanation_mode or "Simple"
    )

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
        "＋ New chat",
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
                    in st.session_state.chats[chat_id][
                        "paper_name"
                    ].lower()
                )
            ]

    if recent_ids:
        for chat_id in recent_ids:
            chat_button(chat_id, "recent")
    else:
        st.caption("No chats yet. Upload a paper to begin.")

    # Current chat actions
    current = get_current_chat()

    if current:
        st.caption("Chat actions")

        is_pinned = current["id"] in st.session_state.pinned

        st.button(
            "Unpin chat" if is_pinned else "Pin chat",
            key="act_pin",
            on_click=toggle_pin,
            args=(current["id"],),
            use_container_width=True,
        )

        st.button(
            "Clear messages",
            key="act_clear",
            on_click=clear_chat,
            args=(current["id"],),
            use_container_width=True,
        )

        if (
            st.session_state.confirm_delete
            == current["id"]
        ):
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
                on_click=set_confirm_delete,
                args=(current["id"],),
                use_container_width=True,
            )


# ============================================================
# 15. MAIN: UPLOAD SCREEN
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
        'Upload a PDF and chat with it. '
        'Answers include references to source pages.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload research paper",
        type=["pdf"],
        key=f"pdf_upload_{st.session_state.upload_key}",
    )

    if uploaded_file:
        with st.spinner(
            "Reading and indexing the research paper..."
        ):
            try:
                file_bytes = uploaded_file.getvalue()

                pages, collection = process_pdf(file_bytes)

                create_chat(
                    uploaded_file.name,
                    collection,
                    pages,
                )

                st.rerun()

            except Exception as e:
                st.error(f"Unable to process PDF: {e}")


# ============================================================
# 16. MAIN: CHAT SCREEN
# ============================================================

else:

    pages = current_chat["pages"]
    collection = current_chat["collection"]
    history = current_chat["history"]

    mode = (
        st.session_state.explanation_mode or "Simple"
    )

    # Header
    total_words = sum(
        len(page["text"].split())
        for page in pages
    )

    title_col, summary_col = st.columns([4, 1])

    with title_col:
        st.markdown(
            (
                '<p class="pl-title">'
                f'{html.escape(current_chat["paper_name"])}'
                '</p>'
                '<p class="pl-meta">'
                f'{len(pages)} pages · '
                f'{total_words:,} words · '
                f'{mode} answers'
                '</p>'
            ),
            unsafe_allow_html=True,
        )

    with summary_col:
        st.button(
            "Summarize paper",
            key="top_summary",
            on_click=ask_summary,
            use_container_width=True,
        )

    # Empty state and suggestions
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

    # Render chat history
    for item in history:

        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            metadata = item.get("metadata", [])
            chunks = item.get("chunks", [])
            distances = item.get("distances", [])

            cited_pages = sorted({
                meta.get("page")
                for meta in metadata
                if meta.get("page")
            })

            if cited_pages:
                page_text = ", ".join(
                    str(page) for page in cited_pages
                )

                st.markdown(
                    (
                        '<div class="pl-cites">'
                        f'Source pages: {page_text}'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

            if chunks:
                with st.expander(
                    f"Sources ({len(chunks)} passages)"
                ):
                    for i, chunk in enumerate(chunks):
                        page_number = (
                            metadata[i].get("page", "?")
                            if i < len(metadata)
                            else "?"
                        )

                        distance = (
                            distances[i]
                            if i < len(distances)
                            else None
                        )

                        if distance is not None:
                            st.markdown(
                                f"**Page {page_number}** "
                                f"· distance {distance:.3f}"
                            )
                        else:
                            st.markdown(
                                f"**Page {page_number}**"
                            )

                        st.caption(chunk)

    # Process pending question or summary
    if pending:

        with st.chat_message("user"):
            st.write(pending["question"])

        with st.chat_message("assistant"):

            with st.spinner(
                "Reading the paper and generating your answer..."
            ):

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

    # Chat input
    question = st.chat_input(
        "Ask about this paper..."
    )

    if question and question.strip():

        st.session_state.pending = {
            "kind": "question",
            "question": question.strip(),
        }

        st.rerun()
