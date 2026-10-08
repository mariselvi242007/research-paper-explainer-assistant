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

SUMMARY_FIELDS = [
    ("objective", "Research objective"),
    ("problem", "Problem statement"),
    ("method", "Methodology"),
    ("dataset", "Dataset"),
    ("technologies", "Algorithms and technologies"),
    ("results", "Main results"),
    ("limitations", "Limitations"),
    ("conclusion", "Conclusion"),
]


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --pl-accent: #2F5D8A;
    --pl-ink: #1B2430;
    --pl-mute: #66758a;
    --pl-line: #D5DDE6;
    --pl-card: #FFFFFF;
    --pl-hl: #FFF3C4;
}

/* ---------- Base ---------- */
html, body, .stApp, .stMarkdown, .stButton button, .stChatInput textarea,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, label, input {
    font-family: 'Public Sans', system-ui, sans-serif;
}
footer, [data-testid="stDecoration"] {display: none;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.4rem; padding-bottom: 1rem; max-width: 1640px;}

/* ---------- Buttons ---------- */
.stButton button {
    border-radius: 10px;
    font-weight: 500;
    transition: background .15s, border-color .15s;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.06);}
.pl-logo {display: flex; align-items: center; gap: .65rem; margin: .2rem 0 1rem 0;}
.pl-logo-mark {
    width: 34px; height: 34px; border-radius: 9px; background: #8DBDEB; color: #0F1B2D;
    display: flex; align-items: center; justify-content: center;
    font: 700 1.05rem 'Newsreader', Georgia, serif;
}
.pl-logo-name {font: 600 1.25rem 'Newsreader', Georgia, serif; line-height: 1.1;}
.pl-logo-sub {font-size: .76rem; opacity: .6;}
.pl-section {font-size: .78rem; font-weight: 600; opacity: .55; margin: 1.1rem 0 .35rem .15rem; letter-spacing: .02em;}

[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: transparent; border: 1px solid transparent; justify-content: flex-start; text-align: left;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,.07); border-color: rgba(255,255,255,.08);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: rgba(141,189,235,.16); color: #fff; border: 1px solid rgba(141,189,235,.4);
    justify-content: flex-start; text-align: left;
}
.st-key-new_chat_btn button {
    background: #8DBDEB !important; color: #0F1B2D !important; font-weight: 600;
    border: none !important; justify-content: center !important;
}
.st-key-new_chat_btn button:hover {background: #A6CDF1 !important;}

/* ---------- Header chips ---------- */
.pl-title {font: 600 1.7rem 'Newsreader', Georgia, serif; margin: 0; line-height: 1.2; color: var(--pl-ink);}
.pl-chips {display: flex; gap: .45rem; flex-wrap: wrap; margin: .5rem 0 .9rem 0;}
.pl-chip {
    background: #fff; border: 1px solid var(--pl-line); border-radius: 99px;
    padding: .15rem .75rem; font-size: .8rem; color: var(--pl-mute);
}
.pl-chip.accent {background: var(--pl-accent); border-color: var(--pl-accent); color: #fff;}

/* ---------- Containers as cards ---------- */
.st-key-viewer_box, .st-key-assist_box {
    background: var(--pl-card);
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(20,35,60,.06), 0 6px 20px rgba(20,35,60,.05);
}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}

/* ---------- Paper viewer ---------- */
.pl-page-tag {
    font-size: .8rem; color: var(--pl-mute); border-bottom: 1px solid var(--pl-line);
    padding-bottom: .4rem; margin-bottom: .6rem;
}
.pl-para {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.08rem; line-height: 1.75; color: var(--pl-ink);
    padding: .4rem .85rem; border-left: 3px solid transparent; border-radius: 4px;
    transition: background .15s, border-color .15s;
}
.pl-para:hover {background: var(--pl-hl); border-left-color: #E0B43C;}
[class*="st-key-explain_"] button {
    font-size: .8rem; padding: .15rem .5rem; min-height: 2rem;
    background: #fff; color: var(--pl-accent); border: 1px solid var(--pl-line);
}
[class*="st-key-explain_"] button:hover {border-color: var(--pl-accent); background: #F2F7FC;}

/* ---------- Segmented control ---------- */
[data-testid="stSegmentedControl"] {width: 100%;}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: #F4F7FA; border: 1px solid #E3E9F0; border-radius: 12px; padding: .7rem .9rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #E8F0F8; border-color: #D3E1EF;
}
[data-testid="stChatInput"] {border-radius: 14px;}
.pl-quote {
    border-left: 3px solid #E0B43C; background: var(--pl-hl); border-radius: 4px;
    padding: .35rem .75rem; font: italic .92rem 'Newsreader', Georgia, serif;
    color: #4b4430; margin-bottom: .5rem;
}
.pl-empty {text-align: center; padding: 1.2rem .5rem .6rem .5rem;}
.pl-empty b {font: 600 1.15rem 'Newsreader', Georgia, serif;}
.pl-empty span {display: block; color: var(--pl-mute); font-size: .88rem; margin-top: .2rem;}

/* ---------- Overview & glossary cards ---------- */
.pl-oneline {
    background: var(--pl-accent); color: #fff; border-radius: 12px; padding: .9rem 1.1rem;
    font: 500 1.05rem/1.5 'Newsreader', Georgia, serif; margin-bottom: .8rem;
}
.pl-field {border: 1px solid var(--pl-line); border-radius: 12px; padding: .65rem .9rem; margin-bottom: .55rem;}
.pl-field-label {font-size: .78rem; font-weight: 600; color: var(--pl-accent); margin-bottom: .15rem;}
.pl-field-text {font-size: .93rem; line-height: 1.55; color: var(--pl-ink);}
.pl-term {border-bottom: 1px solid var(--pl-line); padding: .55rem .2rem;}
.pl-term b {font-size: .95rem;}
.pl-term span {display: block; color: var(--pl-mute); font-size: .88rem; margin-top: .1rem;}

/* ---------- Upload screen ---------- */
.pl-hero-title {font: 600 3rem 'Newsreader', Georgia, serif; line-height: 1.1; margin: 1.5rem 0 .4rem 0; color: var(--pl-ink);}
.pl-hero-sub {font-size: 1.05rem; color: var(--pl-mute); max-width: 36rem; margin-bottom: 1.4rem;}
[data-testid="stFileUploaderDropzone"] {
    background: #fff; border: 2px dashed #9DB6CF; border-radius: 16px; padding: 2rem 1.2rem;
}
[data-testid="stFileUploaderDropzone"]:hover {border-color: var(--pl-accent); background: #F7FAFD;}
.pl-feature {
    background: #fff; border: 1px solid var(--pl-line); border-radius: 14px; padding: .9rem 1rem; height: 100%;
}
.pl-feature b {display: block; margin-bottom: .2rem; font-size: .95rem;}
.pl-feature span {color: var(--pl-mute); font-size: .86rem; line-height: 1.45;}

@media (prefers-reduced-motion: reduce) {* {transition: none !important;}}
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


def split_into_paragraphs(text):
    """Break a page into readable blocks for the paper viewer."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    if len(blocks) < 3:
        flat = text.replace("\n", " ")
        sentences = re.split(r"(?<=[.!?])\s+", flat)
        blocks = [" ".join(sentences[i:i + 3]) for i in range(0, len(sentences), 3)]

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


def step_page(delta, page_numbers):
    current = st.session_state.get("viewer_page", page_numbers[0])
    index = page_numbers.index(current) if current in page_numbers else 0
    index = max(0, min(len(page_numbers) - 1, index + delta))
    st.session_state.viewer_page = page_numbers[index]


# ============================================================
# AI FUNCTIONS
# ============================================================

def paper_text_for_prompt(pages, limit=60000):
    text = "\n\n".join(f"PAGE {p['page']}\n{p['text']}" for p in pages)
    return text[:limit]


def parse_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    return json.loads(cleaned)


def generate_summary(pages):
    prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research paper provided below.
Do not use outside knowledge. Do not invent information.

Return a JSON object with exactly these string keys:
"one_line" (one sentence that captures the whole paper),
"objective", "problem", "method", "dataset", "technologies",
"results", "limitations", "conclusion".

Each value is 1 to 3 short sentences. If information is not available in
the paper, use: "Not specified in the paper."

Research paper:
{paper_text_for_prompt(pages)}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return parse_json(response.text)


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
    data = parse_json(response.text)
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
    title = st.session_state.chats[chat_id]["title"]
    if len(title) > 30:
        title = title[:27] + "..."

    is_current = chat_id == st.session_state.current_chat_id

    if st.button(
        title,
        key=f"{prefix}_{chat_id}",
        use_container_width=True,
        type="primary" if is_current else "secondary",
        icon=":material/description:",
    ):
        open_chat(chat_id)
        st.rerun()


with st.sidebar:
    st.markdown(
        """
<div class="pl-logo">
    <div class="pl-logo-mark">P</div>
    <div>
        <div class="pl-logo-name">PaperLens</div>
        <div class="pl-logo-sub">Research paper assistant</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "New chat",
        key="new_chat_btn",
        use_container_width=True,
        icon=":material/add:",
    ):
        new_chat()
        st.rerun()

    st.markdown('<div class="pl-section">Explanation level</div>', unsafe_allow_html=True)
    st.segmented_control(
        "Explanation level",
        LEVELS,
        key="explanation_mode",
        label_visibility="collapsed",
    )

    pinned_ids = [c for c in st.session_state.pinned if c in st.session_state.chats]
    if pinned_ids:
        st.markdown('<div class="pl-section">Pinned</div>', unsafe_allow_html=True)
        for chat_id in pinned_ids:
            chat_button(chat_id, "pinned")

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
        st.caption("No recent chats")

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
    _, center, _ = st.columns([1, 3, 1])

    with center:
        st.markdown('<div class="pl-hero-title">Read any paper<br>with a guide beside you.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pl-hero-sub">Upload a research paper, read it on the left, '
            "and get explanations on the right. Every answer points back to the page it came from.</div>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload research paper",
            type=["pdf"],
            key=f"pdf_upload_{st.session_state.upload_key}",
            label_visibility="collapsed",
        )

        st.write("")
        f1, f2, f3 = st.columns(3)
        features = [
            ("Explain any paragraph", "Press Explain next to a passage and get it in plain words."),
            ("Choose your level", "Switch between Simple, Student and Expert at any time."),
            ("Answers with page numbers", "Jump straight to the source page behind each answer."),
        ]
        for col, (head, body) in zip((f1, f2, f3), features):
            col.markdown(
                f'<div class="pl-feature"><b>{head}</b><span>{body}</span></div>',
                unsafe_allow_html=True,
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
    level = st.session_state.explanation_mode or "Simple"

    # ---------------- Top bar ----------------

    title_col, button_col = st.columns([6, 1])

    with title_col:
        total_words = sum(len(p["text"].split()) for p in pages)
        st.markdown(f'<p class="pl-title">{html.escape(paper_name)}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pl-chips">'
            f'<span class="pl-chip">{len(pages)} pages</span>'
            f'<span class="pl-chip">{total_words:,} words</span>'
            f'<span class="pl-chip accent">{level} explanations</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with button_col:
        if st.button("New paper", use_container_width=True, icon=":material/upload_file:"):
            new_chat()
            st.rerun()

    viewer_col, assistant_col = st.columns([3, 2], gap="large")

    # ---------------- Left: paper viewer ----------------

    with viewer_col:
        page_numbers = [p["page"] for p in pages]
        page_lookup = {p["page"]: p["text"] for p in pages}

        if st.session_state.get("viewer_page") not in page_numbers:
            st.session_state.viewer_page = page_numbers[0]

        with st.container(key="viewer_box", border=True):
            prev_col, select_col, next_col = st.columns([1, 5, 1])

            with prev_col:
                st.button(
                    "Prev",
                    key="prev_page",
                    on_click=step_page,
                    args=(-1, page_numbers),
                    use_container_width=True,
                    icon=":material/chevron_left:",
                )

            with select_col:
                st.selectbox(
                    "Page",
                    page_numbers,
                    key="viewer_page",
                    format_func=lambda p: f"Page {p} of {page_numbers[-1]}",
                    label_visibility="collapsed",
                )

            with next_col:
                st.button(
                    "Next",
                    key="next_page",
                    on_click=step_page,
                    args=(1, page_numbers),
                    use_container_width=True,
                    icon=":material/chevron_right:",
                )

            page = st.session_state.viewer_page

            with st.container(height=560, border=False):
                st.markdown(
                    '<div class="pl-page-tag">Hover a paragraph, then press Explain to get it explained.</div>',
                    unsafe_allow_html=True,
                )

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
                            icon=":material/lightbulb:",
                        )

    # ---------------- Right: assistant panel ----------------

    with assistant_col:
        st.segmented_control(
            "Panel",
            ["Chat", "Overview", "Glossary"],
            key="panel",
            label_visibility="collapsed",
        )

        panel = st.session_state.panel or "Chat"

        # ----- Chat -----
        if panel == "Chat":
            pending = st.session_state.pending

            with st.container(key="assist_box", height=520, border=True):
                if not history and not pending:
                    st.markdown(
                        '<div class="pl-empty"><b>Ask anything about this paper</b>'
                        "<span>Or start with one of these</span></div>",
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

                for m_idx, item in enumerate(history):
                    with st.chat_message("user", avatar=":material/person:"):
                        if item.get("passage"):
                            passage = item["passage"]
                            st.markdown(
                                f'<div class="pl-quote">{html.escape(passage[:300])}'
                                f'{"..." if len(passage) > 300 else ""}</div>',
                                unsafe_allow_html=True,
                            )
                        st.write(item["question"])

                    with st.chat_message("assistant", avatar=":material/auto_stories:"):
                        st.write(item["answer"])

                        metadata = item.get("metadata", [])
                        chunks = item.get("chunks", [])
                        distances = item.get("distances", [])

                        cited = sorted({m.get("page") for m in metadata if m.get("page")})[:4]
                        if cited:
                            cols = st.columns(len(cited))
                            for c, p in zip(cols, cited):
                                c.button(
                                    f"Page {p}",
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
                    with st.chat_message("user", avatar=":material/person:"):
                        if pending.get("passage"):
                            st.markdown(
                                f'<div class="pl-quote">{html.escape(pending["passage"][:300])}</div>',
                                unsafe_allow_html=True,
                            )
                        st.write(pending["question"])

                    with st.chat_message("assistant", avatar=":material/auto_stories:"):
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
            with st.container(key="assist_box", height=580, border=True):
                summary = current_chat["summary"]

                if summary:
                    if summary.get("one_line"):
                        st.markdown(
                            f'<div class="pl-oneline">{html.escape(str(summary["one_line"]))}</div>',
                            unsafe_allow_html=True,
                        )

                    for field, label in SUMMARY_FIELDS:
                        value = summary.get(field)
                        if value:
                            st.markdown(
                                f'<div class="pl-field"><div class="pl-field-label">{label}</div>'
                                f'<div class="pl-field-text">{html.escape(str(value))}</div></div>',
                                unsafe_allow_html=True,
                            )

                    if st.button("Regenerate overview", icon=":material/refresh:"):
                        current_chat["summary"] = None
                        st.rerun()
                else:
                    st.markdown(
                        '<div class="pl-empty"><b>Paper at a glance</b>'
                        "<span>Objective, method, dataset, results and limitations in one view.</span></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Generate overview", type="primary", use_container_width=True):
                        with st.spinner("Generating overview..."):
                            try:
                                current_chat["summary"] = generate_summary(pages)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Unable to generate overview: {e}")

        # ----- Glossary -----
        else:
            with st.container(key="assist_box", height=580, border=True):
                glossary = current_chat["glossary"]

                if glossary:
                    for entry in glossary:
                        st.markdown(
                            f'<div class="pl-term"><b>{html.escape(entry["term"])}</b>'
                            f'<span>{html.escape(entry["meaning"])}</span></div>',
                            unsafe_allow_html=True,
                        )

                    st.write("")
                    if st.button("Regenerate glossary", icon=":material/refresh:"):
                        current_chat["glossary"] = None
                        st.rerun()
                else:
                    st.markdown(
                        '<div class="pl-empty"><b>Key terms</b>'
                        "<span>Important terms from this paper, in plain language.</span></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Generate glossary", type="primary", use_container_width=True):
                        with st.spinner("Collecting key terms..."):
                            try:
                                current_chat["glossary"] = generate_glossary(pages)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Unable to generate glossary: {e}")
