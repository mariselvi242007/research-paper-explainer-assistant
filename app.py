import html
import json
import re
import uuid
import hashlib
from io import BytesIO

import streamlit as st
import chromadb
from google import genai
from google.genai import types
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
LEVELS = ["Simple", "Student", "Expert"]

STYLE_INSTRUCTIONS = {
    "Simple": (
        "Explain in plain, beginner-friendly English. Avoid jargon, and if a "
        "technical term is unavoidable, explain it in a few words."
    ),
    "Student": (
        "Explain clearly for a university student. Use standard terminology "
        "but define key terms briefly."
    ),
    "Expert": (
        "Give a precise, technical explanation using proper research "
        "terminology. Be concise and detailed where it matters."
    ),
}


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>
    footer {visibility: hidden;}
    .block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1600px;}

    /* Sidebar */
    [data-testid="stSidebar"] .stButton button {
        justify-content: flex-start;
        text-align: left;
        border-radius: 8px;
    }
    .pl-brand {font-size: 1.35rem; font-weight: 700; margin-bottom: 0;}
    .pl-sub {opacity: .65; font-size: .85rem; margin-bottom: .5rem;}
    .pl-label {font-size: .78rem; font-weight: 600; opacity: .65; margin: .9rem 0 .3rem 0;}

    /* Paper viewer */
    .pl-para {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.02rem;
        line-height: 1.7;
        padding: .35rem .8rem;
        border-left: 3px solid rgba(128,128,128,.28);
        margin-bottom: .2rem;
    }

    /* Panels */
    .pl-quote {
        border-left: 3px solid #e0b43c;
        padding: .15rem .8rem;
        opacity: .8;
        font-style: italic;
        margin-bottom: .6rem;
        font-size: .92rem;
    }
    .pl-title {font-size: 1.3rem; font-weight: 650; margin: 0;}
    .pl-meta {opacity: .65; font-size: .88rem; margin-bottom: .2rem;}
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
    "panel": "Chat",
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
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return f"research_paper_{file_hash[:10]}"


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


def split_into_paragraphs(text):
    """Break a page into readable blocks for the paper viewer."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    if len(blocks) < 3:
        flat = text.replace("\n", " ")
        sentences = re.split(r"(?<=[.!?])\s+", flat)
        blocks = [
            " ".join(sentences[i:i + 3])
            for i in range(0, len(sentences), 3)
        ]

    return [" ".join(b.split()) for b in blocks if b.strip()]


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
        "summary": None,
        "glossary": None,
        "history": [],
    }

    st.session_state.recents.insert(0, chat_id)
    st.session_state.current_chat_id = chat_id
    st.session_state.panel = "Chat"
    st.session_state.pending = None
    return chat_id


def get_current_chat():
    chat_id = st.session_state.current_chat_id
    return st.session_state.chats.get(chat_id) if chat_id else None


def open_chat(chat_id):
    if chat_id not in st.session_state.chats:
        return

    st.session_state.current_chat_id = chat_id
    st.session_state.pending = None

    if chat_id in st.session_state.recents:
        st.session_state.recents.remove(chat_id)
    st.session_state.recents.insert(0, chat_id)


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


# ============================================================
# CALLBACKS (run before the page re-renders)
# ============================================================

def explain_passage(text, page):
    st.session_state.pending = {
        "question": f"Explain this passage from page {page}.",
        "passage": text,
        "page": page,
    }
    st.session_state.panel = "Chat"


def ask_suggestion(question):
    st.session_state.pending = {"question": question, "passage": None, "page": None}
    st.session_state.panel = "Chat"


def go_to_page(page):
    st.session_state.viewer_page = page


# ============================================================
# AI FUNCTIONS
# ============================================================

def paper_text_for_prompt(pages, limit=60000):
    text = "\n\n".join(f"PAGE {p['page']}\n{p['text']}" for p in pages)
    return text[:limit]


def generate_summary(pages):
    prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research paper provided below.
Do not use outside knowledge. Do not invent information.

Write the answer in Markdown using EXACTLY these sections, each as a
bold heading followed by 1 to 3 short sentences or bullets:

**In one line**
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
{paper_text_for_prompt(pages)}
"""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


def generate_glossary(pages):
    prompt = f"""
From ONLY the research paper below, list 8 to 12 important technical terms
a newcomer might not know, each with a short plain-English meaning (max 25
words) based on how the paper uses it.

Return a JSON array of objects with the keys "term" and "meaning".

Research paper:
{paper_text_for_prompt(pages, limit=40000)}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    text = re.sub(r"^```(?:json)?|```$", "", response.text.strip()).strip()
    data = json.loads(text)
    return [
        {"term": str(d.get("term", "")), "meaning": str(d.get("meaning", ""))}
        for d in data
        if d.get("term")
    ]


def ask_question(question, collection, level, passage=None):
    query_text = passage if passage else question
    query_embedding = embedding_model.encode([query_text])[0]

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

    passage_block = (
        f"\nThe user selected this passage and wants it explained:\n\"\"\"\n{passage}\n\"\"\"\n"
        if passage
        else ""
    )

    prompt = f"""
You are PaperLens, a research paper question-answering assistant.

Answer using ONLY the retrieved content from the research paper.
Do NOT use outside knowledge. Do NOT invent information.
Mention page numbers like (p. 3) when you use a source.

{STYLE_INSTRUCTIONS[level]}

If the answer cannot be found in the retrieved paper content, say exactly:
The information is not available in the research paper.

User request:
{question}
{passage_block}
Retrieved paper content:
{context}
"""

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text, chunks, metadata, distances


# ============================================================
# SIDEBAR
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
    ):
        open_chat(chat_id)
        st.rerun()


with st.sidebar:
    st.markdown('<p class="pl-brand">PaperLens</p>', unsafe_allow_html=True)
    st.markdown('<p class="pl-sub">Research paper assistant</p>', unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True, icon=":material/add:"):
        new_chat()
        st.rerun()

    st.markdown('<p class="pl-label">Explanation level</p>', unsafe_allow_html=True)
    st.radio(
        "Explanation level",
        LEVELS,
        key="explanation_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    pinned_ids = [c for c in st.session_state.pinned if c in st.session_state.chats]
    if pinned_ids:
        st.markdown('<p class="pl-label">Pinned</p>', unsafe_allow_html=True)
        for chat_id in pinned_ids:
            chat_button(chat_id, "pinned")

    st.markdown('<p class="pl-label">Recents</p>', unsafe_allow_html=True)
    recent_ids = [
        c
        for c in st.session_state.recents
        if c in st.session_state.chats and c not in st.session_state.pinned
    ]
    if recent_ids:
        for chat_id in recent_ids:
            chat_button(chat_id, "recent")
    else:
        st.caption("No recent chats")

    current = get_current_chat()
    if current:
        st.divider()
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
    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.write("")
        st.write("")
        st.markdown("# PaperLens")
        st.markdown(
            "Upload a research paper, read it on the left, and get explanations "
            "on the right, with page references for every answer."
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
# MAIN: PAPER WORKSPACE
# ============================================================

else:
    paper_name = current_chat["paper_name"]
    pages = current_chat["pages"]
    collection = current_chat["collection"]
    history = current_chat["history"]
    level = st.session_state.explanation_mode

    # ---------------- Top bar ----------------

    title_col, button_col = st.columns([6, 1])

    with title_col:
        total_words = sum(len(p["text"].split()) for p in pages)
        st.markdown(f'<p class="pl-title">{html.escape(paper_name)}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="pl-meta">{len(pages)} pages · {total_words:,} words · '
            f"{level} explanations</p>",
            unsafe_allow_html=True,
        )

    with button_col:
        if st.button("New paper", use_container_width=True):
            new_chat()
            st.rerun()

    viewer_col, assistant_col = st.columns([3, 2], gap="large")

    # ---------------- Left: paper viewer ----------------

    with viewer_col:
        page_numbers = [p["page"] for p in pages]
        page_lookup = {p["page"]: p["text"] for p in pages}

        if st.session_state.get("viewer_page") not in page_numbers:
            st.session_state.viewer_page = page_numbers[0]

        nav_col, hint_col = st.columns([1, 2])

        with nav_col:
            st.selectbox(
                "Page",
                page_numbers,
                key="viewer_page",
                format_func=lambda p: f"Page {p} of {page_numbers[-1]}",
                label_visibility="collapsed",
            )

        with hint_col:
            st.caption("Press Explain next to any paragraph to get it explained.")

        page = st.session_state.viewer_page

        with st.container(height=600, border=True):
            for idx, block in enumerate(split_into_paragraphs(page_lookup[page])):
                text_col, action_col = st.columns([11, 2])

                with text_col:
                    safe = html.escape(block).replace("$", "&#36;")
                    st.markdown(f'<div class="pl-para">{safe}</div>', unsafe_allow_html=True)

                with action_col:
                    st.button(
                        "Explain",
                        key=f"explain_{page}_{idx}",
                        on_click=explain_passage,
                        args=(block, page),
                        use_container_width=True,
                    )

    # ---------------- Right: assistant panel ----------------

    with assistant_col:
        st.radio(
            "Panel",
            ["Chat", "Overview", "Glossary"],
            key="panel",
            horizontal=True,
            label_visibility="collapsed",
        )

        panel = st.session_state.panel

        # ----- Chat -----
        if panel == "Chat":
            pending = st.session_state.pending

            with st.container(height=520, border=True):
                if not history and not pending:
                    st.markdown("**Ask anything about this paper**")
                    st.caption("Or start with one of these:")
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

                for m_idx, item in enumerate(history):
                    with st.chat_message("user"):
                        if item.get("passage"):
                            st.markdown(
                                f'<div class="pl-quote">{html.escape(item["passage"][:300])}'
                                f'{"..." if len(item["passage"]) > 300 else ""}</div>',
                                unsafe_allow_html=True,
                            )
                        st.write(item["question"])

                    with st.chat_message("assistant"):
                        st.write(item["answer"])

                        metadata = item.get("metadata", [])
                        chunks = item.get("chunks", [])
                        distances = item.get("distances", [])

                        cited = sorted({m.get("page") for m in metadata if m.get("page")})[:4]
                        if cited:
                            cols = st.columns(len(cited))
                            for c, p in zip(cols, cited):
                                c.button(
                                    f"p. {p}",
                                    key=f"go_{m_idx}_{p}",
                                    on_click=go_to_page,
                                    args=(p,),
                                    use_container_width=True,
                                )

                        if chunks:
                            with st.expander(f"Sources ({len(chunks)} passages)"):
                                for i, chunk in enumerate(chunks):
                                    st.markdown(
                                        f"**Page {metadata[i].get('page', '?')}**"
                                        f" · distance {distances[i]:.3f}"
                                    )
                                    st.caption(chunk)

                if pending:
                    with st.chat_message("user"):
                        if pending.get("passage"):
                            st.markdown(
                                f'<div class="pl-quote">{html.escape(pending["passage"][:300])}</div>',
                                unsafe_allow_html=True,
                            )
                        st.write(pending["question"])

                    with st.chat_message("assistant"):
                        with st.spinner("Reading the paper..."):
                            try:
                                answer, chunks, metadata, distances = ask_question(
                                    pending["question"],
                                    collection,
                                    level,
                                    pending.get("passage"),
                                )

                                history.append(
                                    {
                                        "question": pending["question"],
                                        "passage": pending.get("passage"),
                                        "answer": answer,
                                        "chunks": chunks,
                                        "metadata": metadata,
                                        "distances": distances,
                                    }
                                )

                                chat_id = current_chat["id"]
                                if chat_id in st.session_state.recents:
                                    st.session_state.recents.remove(chat_id)
                                st.session_state.recents.insert(0, chat_id)

                                st.session_state.pending = None
                                st.rerun()

                            except Exception as e:
                                st.session_state.pending = None
                                st.error(f"Unable to answer the question: {e}")

            question = st.chat_input("Ask about this paper...")

            if question and question.strip():
                st.session_state.pending = {
                    "question": question.strip(),
                    "passage": None,
                    "page": None,
                }
                st.rerun()

        # ----- Overview -----
        elif panel == "Overview":
            with st.container(height=580, border=True):
                if current_chat["summary"]:
                    st.markdown(current_chat["summary"])
                    if st.button("Regenerate overview"):
                        current_chat["summary"] = None
                        st.rerun()
                else:
                    st.markdown("**Paper at a glance**")
                    st.caption(
                        "A structured summary: objective, method, dataset, "
                        "results and limitations."
                    )
                    if st.button("Generate overview", type="primary"):
                        with st.spinner("Generating overview..."):
                            try:
                                current_chat["summary"] = generate_summary(pages)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Unable to generate overview: {e}")

        # ----- Glossary -----
        else:
            with st.container(height=580, border=True):
                if current_chat["glossary"]:
                    for entry in current_chat["glossary"]:
                        st.markdown(f"**{entry['term']}**")
                        st.caption(entry["meaning"])
                    if st.button("Regenerate glossary"):
                        current_chat["glossary"] = None
                        st.rerun()
                else:
                    st.markdown("**Key terms**")
                    st.caption("Important terms from this paper in plain language.")
                    if st.button("Generate glossary", type="primary"):
                        with st.spinner("Collecting key terms..."):
                            try:
                                current_chat["glossary"] = generate_glossary(pages)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Unable to generate glossary: {e}")
