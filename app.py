import html
import uuid
import hashlib
from io import BytesIO

import streamlit as st
import chromadb
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

# If the main model hits its daily free quota, the next one is tried automatically.
# (Gemini free-tier quotas are counted separately for each model.)
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
MODES = ["Simple", "Technical"]


MODE_HINTS = {
    "Simple": "Plain, beginner-friendly answers.",
    "Technical": "Detailed answers with research terminology.",
}

STYLE_INSTRUCTIONS = {
    "Simple": (
        "Explain the answer in simple English. Use beginner-friendly language "
        "and avoid unnecessary technical terminology."
    ),
    "Technical": (
        "Give a technical and detailed explanation. Use appropriate research "
        "and computer science terminology."
    ),
}


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

.stApp {font-family: 'Public Sans', system-ui, sans-serif;}
footer, #MainMenu {display: none;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stToolbar"], [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {display: none !important;}

/* Centered reading column for the chat */
.block-container {max-width: 880px; padding-top: 2rem; padding-bottom: 6rem;}

/* ---------- Sidebar ---------- */
.pl-logo-name {font: 600 1.45rem 'Newsreader', Georgia, serif; line-height: 1.1; margin: 0 0 .3rem 0;}

/* Recent chats: flat rows, title + small subtitle, accent bar on the open chat */
[data-testid="stSidebar"] [class*="st-key-recent_"] button,
[data-testid="stSidebar"] [class*="st-key-pinned_"] button {
    background: transparent; border: none; border-radius: 8px;
    justify-content: flex-start; text-align: left; font-weight: 500;
    padding: .5rem .75rem; min-height: 0; box-shadow: none;
}
[data-testid="stSidebar"] [class*="st-key-recent_"] button:hover,
[data-testid="stSidebar"] [class*="st-key-pinned_"] button:hover {background: rgba(47,93,138,.08);}
[data-testid="stSidebar"] [class*="st-key-recent_"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] [class*="st-key-pinned_"] [data-testid="stBaseButton-primary"] {
    background: #E6EEF7; color: #1B2430; box-shadow: inset 3px 0 0 #2F5D8A;
}
.st-key-new_chat_btn button {
    background: #2F5D8A !important; color: #fff !important; border: none !important;
    justify-content: center !important; font-weight: 600; border-radius: 9px;
}
.st-key-new_chat_btn button:hover {background: #254B72 !important;}


/* ---------- Sidebar spacing ---------- */
[data-testid="stSidebarHeader"] {height: auto; padding: .6rem 1rem 0 1rem;}
[data-testid="stSidebarUserContent"] {padding-top: .3rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .6rem;}

/* Left-align text inside sidebar buttons */
[data-testid="stSidebar"] .stButton button > div {justify-content: flex-start; width: 100%;}
[data-testid="stSidebar"] .stButton button p {text-align: left;}
.st-key-new_chat_btn button > div {justify-content: center !important;}

/* Chat actions: flat rows */
[class*="st-key-act_"] button {
    background: transparent; border: none; box-shadow: none; border-radius: 8px;
    padding: .4rem .75rem; min-height: 0; font-weight: 500; justify-content: flex-start;
}
[class*="st-key-act_"] button:hover {background: rgba(47,93,138,.08);}
.st-key-act_delete button, .st-key-act_delete_yes button {color: #B3382C;}
.st-key-act_delete button:hover {background: rgba(179,56,44,.08);}
.st-key-act_delete_yes button, .st-key-act_delete_no button {
    justify-content: center !important; border: 1px solid #D5DDE6 !important;
}
.st-key-act_delete_yes button > div, .st-key-act_delete_no button > div {justify-content: center !important;}

/* ---------- Header ---------- */
.pl-title {font: 600 1.55rem 'Newsreader', Georgia, serif; margin: 0; line-height: 1.25;}
.pl-meta {color: #66758a; font-size: .85rem; margin: .2rem 0 .8rem 0;}

/* ---------- Chat ---------- */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {display: none;}
[data-testid="stChatMessage"] {
    width: fit-content; max-width: 88%; border-radius: 16px; padding: .8rem 1.1rem;
    background: #F4F7FA; border: 1px solid #E3E9F0; margin-right: auto;
}
/* Questions: right side */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto; margin-right: 0; max-width: 72%;
    background: #E1ECF7; border-color: #C9DCEF; border-bottom-right-radius: 4px;
}
/* Answers: left side */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-bottom-left-radius: 4px; background: #FFFFFF; border-color: #DDE5EE;
}
.pl-cites {color: #66758a; font-size: .82rem; margin-top: .4rem;}
.pl-empty {text-align: center; padding: 3rem 1rem 1rem 1rem;}
.pl-empty b {font: 600 1.5rem 'Newsreader', Georgia, serif;}
.pl-empty span {display: block; color: #66758a; margin-top: .3rem;}

/* ---------- Upload screen ---------- */
.pl-hero-title {font: 600 2.8rem 'Newsreader', Georgia, serif; line-height: 1.12; margin: 2rem 0 .5rem 0;}
.pl-hero-sub {font-size: 1.05rem; color: #66758a; margin-bottom: 1.4rem;}
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #9DB6CF; border-radius: 16px; padding: 2rem 1.2rem; background: #F7FAFD;
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
# GEMINI CLIENT
# ============================================================

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()


# ============================================================
# MODELS
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
            pages.append({"page": page_number, "text": text})

    return pages


def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, page_numbers = [], []

    for page_data in pages:
        for chunk in splitter.split_text(page_data["text"]):
            chunks.append(chunk)
            page_numbers.append(page_data["page"])

    return chunks, page_numbers


def create_collection_name(file_bytes):
    return f"research_paper_{hashlib.md5(file_bytes).hexdigest()[:10]}"


def process_pdf(file_bytes):
    pages = extract_pdf_pages(file_bytes)
    if not pages:
        raise ValueError("No readable text was found in this PDF.")

    chunks, page_numbers = create_chunks(pages)
    if not chunks:
        raise ValueError("No text chunks were created.")

    embeddings = embedding_model.encode(chunks, show_progress_bar=False)

    collection = chroma_client.get_or_create_collection(
        name=create_collection_name(file_bytes)
    )

    if collection.count() == 0:
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"page": p} for p in page_numbers],
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
    return st.session_state.chats.get(chat_id) if chat_id else None


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
    st.session_state.pending = {"kind": "question", "question": question}


def ask_summary():
    st.session_state.pending = {"kind": "summary", "question": "Summarize this paper."}


# ============================================================
# AI FUNCTIONS
# ============================================================

class QuotaExceeded(Exception):
    """Raised when every configured Gemini model has hit its quota."""


def get_model_candidates():
    try:
        primary = st.secrets.get("GEMINI_MODEL", PRIMARY_MODEL)
    except Exception:
        primary = PRIMARY_MODEL

    models = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    return models


def generate_text(prompt):
    """Call Gemini, moving to the next model if one is out of quota or unavailable."""
    quota_hit = False
    last_error = None

    for model in get_model_candidates():
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text or "The model returned an empty answer. Please try again."
        except Exception as e:
            message = str(e)
            last_error = e

            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                quota_hit = True
                continue

            if any(code in message for code in ("404", "NOT_FOUND", "503", "UNAVAILABLE")):
                continue

            raise

    if quota_hit:
        raise QuotaExceeded()

    raise last_error


def friendly_error(error):
    if isinstance(error, QuotaExceeded):
        return (
            "The Gemini API quota for today is used up on every model. "
            "It resets at midnight Pacific time. To continue now, add billing to your "
            "Google AI Studio project or use an API key from a different project."
        )

    text = str(error)
    if len(text) > 300:
        text = text[:300] + "..."
    return f"Something went wrong while contacting the model: {text}"


def generate_summary(pages):
    paper_text = "\n\n".join(
        f"PAGE {p['page']}\n{p['text']}" for p in pages
    )[:60000]

    prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research paper provided below.
Do not use outside knowledge. Do not invent information.

Write the answer in Markdown using EXACTLY these sections, each as a bold
heading followed by 1 to 3 short sentences or bullets:

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


def ask_question(question, collection, mode):
    query_embedding = embedding_model.encode([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, max(collection.count(), 1)),
    )

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    context = "\n\n".join(
        f"SOURCE {i + 1}\nPAGE: {metadata[i].get('page', 'Unknown')}\n\nTEXT:\n{chunks[i]}"
        for i in range(len(chunks))
    )

    prompt = f"""
You are PaperLens, a research paper question-answering assistant.

Answer using ONLY the retrieved content from the research paper.
Do NOT use outside knowledge. Do NOT invent information.
Mention page numbers like (p. 3) when you use a source.

{STYLE_INSTRUCTIONS[mode]}

If the answer cannot be found in the retrieved paper content, say exactly:
The information is not available in the research paper.

User question:
{question}

Retrieved paper content:
{context}
"""

    return generate_text(prompt), chunks, metadata, distances


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
        help=f"{chat['paper_name']} · {len(chat['history'])} messages",
    ):
        open_chat(chat_id)
        st.rerun()


with st.sidebar:
    st.markdown('<div class="pl-logo-name">PaperLens</div>', unsafe_allow_html=True)

    # Answer style, at the top
    st.caption("Answer style")
    st.segmented_control(
        "Answer style",
        MODES,
        key="explanation_mode",
        label_visibility="collapsed",
    )
    active_mode = st.session_state.explanation_mode or "Simple"
    st.caption(MODE_HINTS[active_mode])

    if st.button("New chat", key="new_chat_btn", use_container_width=True):
        new_chat()
        st.rerun()

    # Pinned
    pinned_ids = [c for c in st.session_state.pinned if c in st.session_state.chats]
    if pinned_ids:
        st.caption("Pinned")
        for chat_id in pinned_ids:
            chat_button(chat_id, "pinned")

    # Recents / history
    st.caption("Recents")

    recent_ids = [
        c
        for c in st.session_state.recents
        if c in st.session_state.chats and c not in st.session_state.pinned
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
                c
                for c in recent_ids
                if search in st.session_state.chats[c]["title"].lower()
                or search in st.session_state.chats[c]["paper_name"].lower()
            ]

    if recent_ids:
        for chat_id in recent_ids:
            chat_button(chat_id, "recent")
    else:
        st.caption("No chats yet. Upload a paper to begin.")

    # Options for the open chat
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
                if st.button("Delete", key="act_delete_yes", use_container_width=True):
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
        '<div class="pl-hero-title">Ask questions about any research paper.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pl-hero-sub">Upload a PDF and chat with it. '
        "Every answer points to the pages it came from.</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload research paper",
        type=["pdf"],
        key=f"pdf_upload_{st.session_state.upload_key}",
        label_visibility="collapsed",
    )

    if uploaded_file:
        with st.spinner("Reading and indexing the paper..."):
            try:
                pages, collection = process_pdf(uploaded_file.getvalue())
                create_chat(uploaded_file.name, collection, pages)
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

    # ---------------- Header ----------------

    total_words = sum(len(p["text"].split()) for p in pages)
    title_col, summary_col = st.columns([4, 1])

    with title_col:
        st.markdown(
            f'<p class="pl-title">{html.escape(current_chat["paper_name"])}</p>'
            f'<p class="pl-meta">{len(pages)} pages · {total_words:,} words · {mode} answers</p>',
            unsafe_allow_html=True,
        )

    with summary_col:
        st.button(
            "Summarize paper",
            key="top_summary",
            on_click=ask_summary,
            use_container_width=True,
        )

    # ---------------- Empty state ----------------

    pending = st.session_state.pending

    if not history and not pending:
        st.markdown(
            '<div class="pl-empty"><b>What would you like to know?</b>'
            "<span>Ask a question below, or try one of these.</span></div>",
            unsafe_allow_html=True,
        )

        for i, suggestion in enumerate(
            [
                "What problem does this paper solve?",
                "Explain the method step by step.",
                "What are the main results and limitations?",
            ]
        ):
            st.button(
                suggestion,
                key=f"suggest_{i}",
                on_click=ask_suggestion,
                args=(suggestion,),
                use_container_width=True,
            )

    # ---------------- History ----------------

    for item in history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

            metadata = item.get("metadata", [])
            chunks = item.get("chunks", [])
            distances = item.get("distances", [])

            cited = sorted({m.get("page") for m in metadata if m.get("page")})
            if cited:
                st.markdown(
                    f'<div class="pl-cites">Pages: {", ".join(str(p) for p in cited)}</div>',
                    unsafe_allow_html=True,
                )

            if chunks:
                with st.expander(f"Sources ({len(chunks)} passages)"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(
                            f"**Page {metadata[i].get('page', '?')}**"
                            f" · distance {distances[i]:.3f}"
                        )
                        st.caption(chunk)

    # ---------------- Pending request ----------------

    if pending:
        with st.chat_message("user"):
            st.write(pending["question"])

        with st.chat_message("assistant"):
            with st.spinner("Reading the paper..."):
                try:
                    if pending["kind"] == "summary":
                        answer = generate_summary(pages)
                        chunks, metadata, distances = [], [], []
                    else:
                        answer, chunks, metadata, distances = ask_question(
                            pending["question"], collection, mode
                        )

                    history.append(
                        {
                            "question": pending["question"],
                            "answer": answer,
                            "chunks": chunks,
                            "metadata": metadata,
                            "distances": distances,
                        }
                    )

                    move_to_top(current_chat["id"])
                    st.session_state.pending = None
                    st.rerun()

                except Exception as e:
                    st.session_state.pending = None
                    if isinstance(e, QuotaExceeded):
                        st.warning(friendly_error(e))
                    else:
                        st.error(friendly_error(e))

    # ---------------- Chat input ----------------

    question = st.chat_input("Ask about this paper...")

    if question and question.strip():
        st.session_state.pending = {"kind": "question", "question": question.strip()}
        st.rerun()
