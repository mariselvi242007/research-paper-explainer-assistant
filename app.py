import streamlit as st
from pypdf import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import hashlib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CHATGPT-STYLE UI
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    html, body, [class*="css"] {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Helvetica,
            Arial,
            sans-serif;
    }

    .stApp {
        background: #212121;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #171717;
        border-right: 1px solid #2f2f2f;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 14px;
        padding-left: 12px;
        padding-right: 12px;
    }

    section[data-testid="stSidebar"] * {
        color: #ececec;
    }

    /* --------------------------------------------------------
       PAPERLENS LOGO
       -------------------------------------------------------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 8px 18px 8px;
    }

    .brand-logo {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: #2f2f2f;
        border: 1px solid #444;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
    }

    .brand-name {
        font-size: 17px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.2px;
    }

    /* --------------------------------------------------------
       SIDEBAR BUTTONS
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 40px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: #ececec;
        text-align: left;
        font-size: 14px;
        font-weight: 400;
        padding: 8px 10px;
        transition: background 0.15s ease;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #2a2a2a;
        border-color: transparent;
    }

    /* New paper button */

    .new-paper-label {
        margin-top: 2px;
        margin-bottom: 10px;
    }

    /* --------------------------------------------------------
       SIDEBAR SECTION LABELS
       -------------------------------------------------------- */

    .side-label {
        color: #9b9b9b;
        font-size: 12px;
        font-weight: 500;
        padding: 15px 9px 6px 9px;
    }

    .current-paper {
        background: #252525;
        border-radius: 8px;
        padding: 9px 10px;
        margin: 4px 0;
    }

    .current-paper-title {
        font-size: 13px;
        color: #eeeeee;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .current-paper-pages {
        font-size: 11px;
        color: #8e8e8e;
        margin-top: 3px;
    }

    /* --------------------------------------------------------
       RADIO
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 3px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 7px;
        padding: 5px 8px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #2a2a2a;
    }

    /* --------------------------------------------------------
       MAIN AREA
       -------------------------------------------------------- */

    .main .block-container {
        max-width: 900px;
        padding-top: 30px;
        padding-bottom: 120px;
    }

    /* --------------------------------------------------------
       WELCOME SCREEN
       -------------------------------------------------------- */

    .welcome {
        text-align: center;
        padding-top: 15vh;
        padding-bottom: 30px;
    }

    .welcome-logo {
        width: 48px;
        height: 48px;
        margin: auto;
        border-radius: 12px;
        background: #2f2f2f;
        border: 1px solid #444;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 23px;
        font-weight: 700;
    }

    .welcome-title {
        margin-top: 18px;
        font-size: 30px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.7px;
    }

    .welcome-subtitle {
        margin-top: 8px;
        font-size: 15px;
        color: #a5a5a5;
    }

    /* --------------------------------------------------------
       PAPER HEADER
       -------------------------------------------------------- */

    .paper-header {
        padding: 10px 0 20px 0;
    }

    .paper-title {
        font-size: 24px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.4px;
    }

    .paper-meta {
        margin-top: 5px;
        font-size: 13px;
        color: #8e8e8e;
    }

    /* --------------------------------------------------------
       CHAT MESSAGES
       -------------------------------------------------------- */

    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 24px 0 18px 0;
    }

    .user-bubble {
        max-width: 75%;
        background: #2f2f2f;
        color: #ececec;
        padding: 11px 15px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.55;
    }

    .assistant-message {
        margin: 20px 0 30px 0;
    }

    .assistant-header {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 9px;
    }

    .assistant-logo {
        width: 27px;
        height: 27px;
        border-radius: 7px;
        background: #303030;
        border: 1px solid #454545;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 13px;
        font-weight: 700;
    }

    .assistant-name {
        color: #eeeeee;
        font-size: 14px;
        font-weight: 600;
    }

    .assistant-content {
        color: #e6e6e6;
        font-size: 15px;
        line-height: 1.7;
    }

    /* --------------------------------------------------------
       SOURCE BOX
       -------------------------------------------------------- */

    .source-title {
        color: #a0a0a0;
        font-size: 12px;
        font-weight: 500;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    /* --------------------------------------------------------
       UPLOADER
       -------------------------------------------------------- */

    div[data-testid="stFileUploader"] {
        margin-top: 10px;
    }

    div[data-testid="stFileUploader"] section {
        background: #2a2a2a;
        border: 1px dashed #555;
        border-radius: 12px;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: #777;
    }

    /* --------------------------------------------------------
       TEXT INPUT
       -------------------------------------------------------- */

    div[data-testid="stTextInput"] input {
        background: #2f2f2f;
        color: #ffffff;
        border: 1px solid #454545;
        border-radius: 12px;
        padding: 12px 15px;
        font-size: 15px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #666;
        box-shadow: none;
    }

    /* --------------------------------------------------------
       PRIMARY BUTTON
       -------------------------------------------------------- */

    .stButton > button[kind="primary"] {
        border-radius: 9px;
        background: #ffffff;
        color: #171717;
        border: none;
        font-weight: 600;
    }

    .stButton > button[kind="primary"]:hover {
        background: #dddddd;
        color: #111111;
    }

    /* --------------------------------------------------------
       EXPANDERS
       -------------------------------------------------------- */

    div[data-testid="stExpander"] {
        background: #252525;
        border: 1px solid #353535;
        border-radius: 9px;
        margin-top: 7px;
    }

    /* --------------------------------------------------------
       DIVIDERS
       -------------------------------------------------------- */

    hr {
        border-color: #343434 !important;
    }

    /* --------------------------------------------------------
       METRICS
       -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: #252525;
        padding: 12px;
        border-radius: 9px;
        border: 1px solid #353535;
    }

    /* --------------------------------------------------------
       ALERTS
       -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 9px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "collection" not in st.session_state:
    st.session_state.collection = None

if "paper_name" not in st.session_state:
    st.session_state.paper_name = None

if "pages" not in st.session_state:
    st.session_state.pages = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "history" not in st.session_state:
    st.session_state.history = []

if "recents" not in st.session_state:
    st.session_state.recents = []

if "explanation_mode" not in st.session_state:
    st.session_state.explanation_mode = "Simple"


# ============================================================
# GEMINI
# ============================================================

try:

    api_key = st.secrets["GEMINI_API_KEY"]

    client = genai.Client(
        api_key=api_key
    )

except Exception:

    client = None


# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_chroma_client():

    return chromadb.Client()


embedding_model = load_embedding_model()
chroma_client = load_chroma_client()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(file_bytes):

    pages = []

    # IMPORTANT:
    # Convert bytes to file-like object.
    # This fixes:
    # "'bytes' object has no attribute 'seek'"

    pdf_file = BytesIO(file_bytes)

    reader = PdfReader(pdf_file)

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:

            page_text = page_text.strip()

        else:

            page_text = ""

        if page_text:

            pages.append(
                {
                    "page": page_number,
                    "text": page_text
                }
            )

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    page_numbers = []

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = text_splitter.split_text(
            page_text
        )

        for chunk in page_chunks:

            chunks.append(chunk)

            page_numbers.append(
                page_number
            )

    return chunks, page_numbers


# ============================================================
# COLLECTION NAME
# ============================================================

def create_collection_name(file_bytes):

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return (
        f"research_paper_{file_hash[:10]}"
    )


# ============================================================
# PROCESS PAPER
# ============================================================

def process_pdf(file_bytes):

    pages = extract_pdf_pages(
        file_bytes
    )

    if not pages:

        raise ValueError(
            "No readable text was found in this PDF."
        )

    chunks, page_numbers = create_chunks(
        pages
    )

    if not chunks:

        raise ValueError(
            "No text chunks were created."
        )

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    collection_name = create_collection_name(
        file_bytes
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    if collection.count() == 0:

        chunk_ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "page": page_numbers[i]
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    return pages, collection


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(pages):

    if client is None:

        raise ValueError(
            "Gemini API key is not configured."
        )

    full_text = "\n\n".join(
        [
            f"[Page {p['page']}]\n{p['text']}"
            for p in pages
        ]
    )

    prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research paper provided below.

Do not use outside knowledge.
Do not invent information.

If something is not available in the paper,
write:

"Not specified in the paper."

Create a clear academic summary using exactly
these sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

Keep the explanation clear and suitable
for a college student.

Research Paper:

{full_text}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def ask_question(
    question,
    collection,
    explanation_mode
):

    if client is None:

        raise ValueError(
            "Gemini API key is not configured."
        )

    # --------------------------------------------------------
    # Question embedding
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )[0]

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=5
    )

    retrieved_chunks = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    retrieved_ids = results["ids"][0]

    if "distances" in results:

        distances = results["distances"][0]

    else:

        distances = [
            0
            for _ in retrieved_chunks
        ]

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = ""

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        page = retrieved_metadata[i].get(
            "page",
            "Unknown"
        )

        context += (
            f"\n\n--- Page {page} ---\n"
            f"{chunk}"
        )

    # --------------------------------------------------------
    # Explanation mode
    # --------------------------------------------------------

    if explanation_mode == "Simple":

        instruction = """
Explain the answer in simple language.

Assume the user is a beginner.

Avoid unnecessary technical jargon.

Use short paragraphs or bullet points
when useful.
"""

    else:

        instruction = """
Give a technical and academically detailed answer.

Use appropriate technical terminology.

Explain algorithms, methodology and concepts
accurately.
"""

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are PaperLens, a research paper
question-answering assistant.

Answer the user's question using ONLY
the retrieved research paper context.

{instruction}

Rules:

- Do not use outside knowledge.
- Do not invent information.
- Do not assume information that is not present.
- Base the answer on the retrieved paper.
- If the answer cannot be found, respond exactly:

"The information is not available in the research paper."

User Question:

{question}

Retrieved Research Paper Context:

{context}
"""

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return (
        response.text,
        retrieved_chunks,
        retrieved_metadata,
        retrieved_ids,
        distances
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="brand">
            <div class="brand-logo">P</div>
            <div class="brand-name">PaperLens</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # NEW PAPER
    # --------------------------------------------------------

    if st.button(
        "＋  New paper",
        use_container_width=True
    ):

        st.session_state.collection = None
        st.session_state.paper_name = None
        st.session_state.pages = []
        st.session_state.summary = None
        st.session_state.history = []

        st.rerun()

    # --------------------------------------------------------
    # RECENTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="side-label">Recents</div>',
        unsafe_allow_html=True
    )

    if st.session_state.recents:

        for index, recent in enumerate(
            reversed(
                st.session_state.recents[-8:]
            )
        ):

            if st.button(
                f"▸  {recent}",
                key=f"recent_{index}_{recent}",
                use_container_width=True
            ):

                st.info(
                    "Upload this paper again to reopen it."
                )

    else:

        st.markdown(
            """
            <div style="
                color:#777;
                font-size:13px;
                padding:6px 9px;
            ">
                No recent papers
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CURRENT PAPER
    # --------------------------------------------------------

    st.markdown(
        '<div class="side-label">Current paper</div>',
        unsafe_allow_html=True
    )

    if st.session_state.paper_name:

        st.markdown(
            f"""
            <div class="current-paper">
                <div class="current-paper-title">
                    {st.session_state.paper_name}
                </div>
                <div class="current-paper-pages">
                    {len(st.session_state.pages)} pages
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div style="
                color:#777;
                font-size:13px;
                padding:6px 9px;
            ">
                No paper uploaded
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="side-label">Response style</div>',
        unsafe_allow_html=True
    )

    st.session_state.explanation_mode = st.radio(
        "Response style",
        ["Simple", "Technical"],
        index=(
            0
            if st.session_state.explanation_mode
            == "Simple"
            else 1
        ),
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    st.markdown("")

    if st.button(
        "Clear session",
        use_container_width=True
    ):

        st.session_state.collection = None
        st.session_state.paper_name = None
        st.session_state.pages = []
        st.session_state.summary = None
        st.session_state.history = []

        st.rerun()

    # --------------------------------------------------------
    # SIDEBAR BOTTOM
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 15px;
            left: 18px;
            color:#777;
            font-size:11px;
        ">
            PaperLens
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.pages:

    # --------------------------------------------------------
    # WELCOME
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="welcome">

            <div class="welcome-logo">
                P
            </div>

            <div class="welcome-title">
                What do you want to understand?
            </div>

            <div class="welcome-subtitle">
                Upload a research paper and explore it
                with PaperLens.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # UPLOADER
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a research paper",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        with st.spinner(
            "Reading your research paper..."
        ):

            try:

                pages, collection = process_pdf(
                    file_bytes
                )

                st.session_state.pages = pages

                st.session_state.collection = (
                    collection
                )

                st.session_state.paper_name = (
                    uploaded_file.name
                )

                st.session_state.summary = None

                st.session_state.history = []

                if (
                    uploaded_file.name
                    not in st.session_state.recents
                ):

                    st.session_state.recents.append(
                        uploaded_file.name
                    )

                st.success(
                    "Paper is ready."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to process PDF: {e}"
                )


# ============================================================
# PAPER LOADED
# ============================================================

else:

    # --------------------------------------------------------
    # PAPER HEADER
    # --------------------------------------------------------

    total_words = sum(
        len(page["text"].split())
        for page in st.session_state.pages
    )

    st.markdown(
        f"""
        <div class="paper-header">

            <div class="paper-title">
                {st.session_state.paper_name}
            </div>

            <div class="paper-meta">
                {len(st.session_state.pages)} pages
                · {total_words:,} words
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SUMMARY BUTTON
    # --------------------------------------------------------

    if st.session_state.summary is None:

        if st.button(
            "Generate paper summary"
        ):

            with st.spinner(
                "Analyzing the research paper..."
            ):

                try:

                    st.session_state.summary = (
                        generate_summary(
                            st.session_state.pages
                        )
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Unable to generate summary: {e}"
                    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if st.session_state.summary:

        with st.expander(
            "Paper summary",
            expanded=True
        ):

            st.markdown(
                st.session_state.summary
            )


    # --------------------------------------------------------
    # PREVIOUS CHAT
    # --------------------------------------------------------

    for item in st.session_state.history:

        # User
        st.markdown(
            f"""
            <div class="user-message">
                <div class="user-bubble">
                    {item["question"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Assistant
        st.markdown(
            """
            <div class="assistant-message">

                <div class="assistant-header">

                    <div class="assistant-logo">
                        P
                    </div>

                    <div class="assistant-name">
                        PaperLens
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            item["answer"]
        )


        # Sources
        if (
            "chunks" in item
            and item["chunks"]
        ):

            st.markdown(
                '<div class="source-title">Sources</div>',
                unsafe_allow_html=True
            )

            for i in range(
                len(item["chunks"])
            ):

                page = item["metadata"][i].get(
                    "page",
                    "Unknown"
                )

                distance = item["distances"][i]

                with st.expander(
                    f"Page {page} · Source {i + 1}"
                ):

                    st.write(
                        item["chunks"][i]
                    )

                    st.caption(
                        f"Relevance distance: {distance:.4f}"
                    )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    st.markdown("---")

    question = st.text_input(
        "Ask about this paper",
        placeholder=(
            "Ask anything about the research paper..."
        ),
        label_visibility="collapsed"
    )

    if st.button(
        "Ask",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Thinking..."
            ):

                try:

                    (
                        answer,
                        retrieved_chunks,
                        metadata,
                        ids,
                        distances
                    ) = ask_question(
                        question,
                        st.session_state.collection,
                        st.session_state.explanation_mode
                    )

                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "chunks": retrieved_chunks,
                            "metadata": metadata,
                            "ids": ids,
                            "distances": distances
                        }
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {e}"
                    )
