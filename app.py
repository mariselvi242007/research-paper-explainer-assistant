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

MODEL_NAME = "gemini-3-flash-preview"
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
footer {display: none;}
header[data-testid="stHeader"] {background: transparent;}

/* Centered reading column for the chat */
.block-container {max-width: 880px; padding-top: 2rem; padding-bottom: 6rem;}

/* ---------- Sidebar ---------- */
.pl-logo {display: flex; align-items: center; gap: .6rem; margin: .1rem 0 .9rem 0;}
.pl-logo-mark {
    width: 32px; height: 32px; border-radius: 9px; background: #2F5D8A; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font: 700 1rem 'Newsreader', Georgia, serif;
}
.pl-logo-name {font: 600 1.2rem 'Newsreader', Georgia, serif; line-height: 1.1;}
.pl-section {font-size: .78rem; font-weight: 600; opacity: .6; margin: 1.1rem 0 .35rem .1rem;}
.pl-hint {font-size: .82rem; opacity: .7; margin-top: .35rem;}

[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: transparent; border: 1px solid transparent;
    justify-content: flex-start; text-align: left; border-radius: 9px;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(47,93,138,.08); border-color: rgba(47,93,138,.15);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #E1ECF7; color: #1B2430; border: 1px solid #B9D0E8;
    justify-content: flex-start; text-align: left; border-radius: 9px;
}
.st-key-new_chat_btn button {
    background: #2F5D8A !important; color: #fff !important; border: none !important;
    justify-content: center !important; font-weight: 600; border-radius: 9px;
}
.st-key-new_chat_btn button:hover {background: #254B72 !important;}

/* ---------- Header ---------- */
.pl-title {font: 600 1.55rem 'Newsreader', Georgia, serif; margin: 0; line-height: 1.25;}
.pl-meta {color: #66758a; font-size: .85rem; margin: .2rem 0 .8rem 0;}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {border-radius: 14px; padding: .75rem 1rem;}
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
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


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

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text, chunks, metadata, distances


# ============================================================
# SIDEBAR
# ============================================================

def chat_button(chat_id, prefix):
    title = st.session_state.chats[chat_id]["title"]
    if len(title) > 34:
        title = title[:31] + "..."

    is_current = chat_id == st.session_state.current_chat_id

    if st.button(
        title,
        key=f"{prefix}_{chat_id}",
        use_container_width=True,
        type="primary" if is_current else "secondary",
    ):
        open_chat(chat_id)
        st.rerun()


with st.sidebar:
    st.markdown(
        '<div class="pl-logo"><div class="pl-logo-mark">P</div>'
        '<div class="pl-logo-name">PaperLens</div></div>',
        unsafe_allow_html=True,
    )

    # Answer style, at the top
    st.markdown('<div class="pl-section">Answer style</div>', unsafe_allow_html=True)
    st.segmented_control(
        "Answer style",
        MODES,
        key="explanation_mode",
        label_visibility="collapsed",
    )
    active_mode = st.session_state.explanation_mode or "Simple"
    st.markdown(f'<div class="pl-hint">{MODE_HINTS[active_mode]}</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("New chat", key="new_chat_btn", use_container_width=True):
        new_chat()
        st.rerun()

    # Pinned
    pinned_ids = [c for c in st.session_state.pinned if c in st.session_state.chats]
    if pinned_ids:
        st.markdown('<div class="pl-section">Pinned</div>', unsafe_allow_html=True)
        for chat_id in pinned_ids:
            chat_button(chat_id, "pinned")

    # Recents / history
    st.markdown('<div class="pl-section">Recents</div>', unsafe_allow_html=True)
    recent_ids = [
        c
        for c in st.session_state.recents
        if c in st.session_state.chats and c not in st.session_state.pinned
    ]
    if recent_ids:
        for chat_id in recent_ids:
            chat_button(chat_id, "recent")
    else:
        st.caption("Your chats will appear here.")

    # Options for the open chat
    current = get_current_chat()
    if current:
        st.markdown('<div class="pl-section">This chat</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            is_pinned = current["id"] in st.session_state.pinned
            if st.button("Unpin" if is_pinned else "Pin", use_container_width=True):
                toggle_pin(current["id"])
                st.rerun()

        with c2:
            if st.button("Clear", use_container_width=True):
                current["history"] = []
                st.rerun()

        with c3:
            if st.button("Delete", use_container_width=True):
                delete_chat(current["id"])
                st.rerun()


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

    title_col, action_col = st.columns([4, 1])

    with title_col:
        total_words = sum(len(p["text"].split()) for p in pages)
        st.markdown(
            f'<p class="pl-title">{html.escape(current_chat["paper_name"])}</p>'
            f'<p class="pl-meta">{len(pages)} pages · {total_words:,} words · {mode} answers</p>',
            unsafe_allow_html=True,
        )

    with action_col:
        st.button(
            "Summarize paper",
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

                    # The first question becomes the chat's title in Recents
                    if not history and pending["kind"] == "question":
                        current_chat["title"] = pending["question"]

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
                    st.error(f"Unable to answer the question: {e}")

    # ---------------- Chat input ----------------

    question = st.chat_input("Ask about this paper...")

    if question and question.strip():
        st.session_state.pending = {"kind": "question", "question": question.strip()}
        st.rerun()
