import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import hashlib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PaperLens | Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown(
    """
    <style>

    /* -------------------------------
       GENERAL PAGE
    --------------------------------*/

    .main {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* -------------------------------
       HEADER
    --------------------------------*/

    .hero {
        padding: 25px 10px 30px 10px;
        text-align: center;
    }

    .hero-icon {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #6b7280;
        max-width: 720px;
        margin: auto;
        line-height: 1.6;
    }


    /* -------------------------------
       SECTION HEADINGS
    --------------------------------*/

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        letter-spacing: -0.3px;
    }

    .section-description {
        color: #6b7280;
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 18px;
    }


    /* -------------------------------
       INFO CARDS
    --------------------------------*/

    .info-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        background: #ffffff;
        min-height: 105px;
    }

    .card-label {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 7px;
    }

    .card-value {
        font-size: 21px;
        font-weight: 700;
    }


    /* -------------------------------
       UPLOAD AREA
    --------------------------------*/

    .upload-card {
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 24px;
        background: #fafafa;
        margin-bottom: 10px;
    }


    /* -------------------------------
       QUESTION AREA
    --------------------------------*/

    .question-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        background: #ffffff;
    }


    /* -------------------------------
       ANSWER AREA
    --------------------------------*/

    .answer-label {
        font-size: 14px;
        font-weight: 700;
        color: #374151;
        margin-bottom: 10px;
    }


    /* -------------------------------
       SOURCE AREA
    --------------------------------*/

    .source-intro {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 12px;
    }


    /* -------------------------------
       SIDEBAR
    --------------------------------*/

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }


    /* -------------------------------
       BUTTONS
    --------------------------------*/

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }


    /* -------------------------------
       FOOTER
    --------------------------------*/

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding-top: 35px;
        margin-top: 40px;
        border-top: 1px solid #eeeeee;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # Brand
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:8px 0 20px 0;
        ">

            <div style="
                font-size:42px;
                margin-bottom:5px;
            ">
                📚
            </div>

            <div style="
                font-size:22px;
                font-weight:750;
                letter-spacing:-0.5px;
            ">
                PaperLens
            </div>

            <div style="
                font-size:12px;
                color:#777;
                margin-top:4px;
            ">
                Research Paper Assistant
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Workspace
    st.markdown(
        """
        <div style="
            font-size:12px;
            font-weight:700;
            color:#6b7280;
            letter-spacing:1px;
            margin-bottom:12px;
        ">
            WORKSPACE
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("📄  Research Paper")
    st.write("📝  Summary")
    st.write("💬  Ask Questions")
    st.write("📚  Sources")

    st.divider()

    # Quick Start
    st.markdown(
        """
        <div style="
            font-size:12px;
            font-weight:700;
            color:#6b7280;
            letter-spacing:1px;
            margin-bottom:12px;
        ">
            QUICK START
        </div>

        <div style="
            font-size:13px;
            line-height:1.9;
            color:#555;
        ">
            <b>01</b>&nbsp;&nbsp;Upload a research paper<br>
            <b>02</b>&nbsp;&nbsp;Generate a summary<br>
            <b>03</b>&nbsp;&nbsp;Ask questions<br>
            <b>04</b>&nbsp;&nbsp;Explore source evidence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # About
    st.markdown(
        """
        <div style="
            font-size:12px;
            font-weight:700;
            color:#6b7280;
            letter-spacing:1px;
            margin-bottom:10px;
        ">
            ABOUT
        </div>

        <div style="
            font-size:12px;
            line-height:1.6;
            color:#777;
        ">
            PaperLens helps students and researchers
            understand academic papers through
            concise summaries and question answering.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            margin-top:25px;
            padding:12px;
            border-radius:9px;
            background:#f8fafc;
            text-align:center;
            font-size:11px;
            color:#777;
        ">
            AI-powered academic assistant
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">
            📚
        </div>

        <div class="hero-title">
            Research Paper Explainer
        </div>

        <div class="hero-subtitle">
            Understand research papers faster with clear summaries,
            intelligent question answering, and evidence from the
            original document.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GEMINI API
# =========================================================

try:

    api_key = st.secrets["GEMINI_API_KEY"]

except Exception:

    st.error(
        "GEMINI_API_KEY is not configured. "
        "Please add it to Streamlit Secrets."
    )

    st.stop()


client = genai.Client(
    api_key=api_key
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# CHROMA CLIENT
# =========================================================

@st.cache_resource
def get_chroma_client():

    return chromadb.Client()


chroma_client = get_chroma_client()


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_pages(file_bytes):

    pages = []

    reader = PdfReader(file_bytes)

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


# =========================================================
# PAGE-AWARE CHUNKING
# =========================================================

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


# =========================================================
# UNIQUE COLLECTION NAME
# =========================================================

def create_collection_name(file_bytes):

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return (
        f"research_paper_{file_hash[:10]}"
    )


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(file_bytes):

    pages = extract_pdf_pages(
        file_bytes
    )

    if not pages:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    chunks, page_numbers = create_chunks(
        pages
    )

    if not chunks:

        raise ValueError(
            "No text chunks were created from the PDF."
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

    # -----------------------------------------------------
    # Add data only if collection is empty
    # -----------------------------------------------------

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


# =========================================================
# GENERATE SUMMARY
# =========================================================

def generate_summary(pages):

    full_text = "\n\n".join(
        [
            f"Page {p['page']}:\n{p['text']}"
            for p in pages
        ]
    )

    # Limit very large papers
    full_text = full_text[:100000]

    prompt = f"""
You are a Research Paper Explainer Assistant.

Summarize ONLY the research paper provided below.

Do not use outside knowledge.

Start directly with:

1. Research Objective

Do not write:
"This summary is based on..."
Do not introduce yourself.
Do not add unnecessary opening text.

Use exactly these sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

Explain every section clearly.

Use simple but technically correct language.

Research Paper:

{full_text}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# =========================================================
# RAG QUESTION ANSWERING
# =========================================================

def ask_question(
    question,
    collection,
    explanation_mode
):

    # -----------------------------------------------------
    # Question embedding
    # -----------------------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )

    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=5
    )

    retrieved_chunks = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    retrieved_ids = results["ids"][0]

    distances = results["distances"][0]

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        page = retrieved_metadata[i].get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
Source {i + 1}
Page: {page}

{chunk}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # -----------------------------------------------------
    # Explanation mode
    # -----------------------------------------------------

    if explanation_mode == "Simple":

        mode_instruction = """
Explain the answer in simple English.

Assume the user is a beginner.

Use short sentences.

Explain technical terms briefly when needed.
"""

    else:

        mode_instruction = """
Explain the answer at a technical level.

Include relevant algorithms,
models, methodology, technical concepts,
and reasoning when available in the paper.
"""

    # -----------------------------------------------------
    # Gemini prompt
    # -----------------------------------------------------

    prompt = f"""
You are a Research Paper Explainer Assistant.

Answer the user's question using ONLY
the research paper context provided below.

Do not use outside knowledge.

{mode_instruction}

Do not invent information.

If the information is not available
in the research paper, say:

"The information is not available in the research paper."

Research Paper Context:

{context}

User Question:

{question}
"""

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


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">Upload Research Paper</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload a PDF to begin analysing and exploring the research paper.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    new_paper = (
        st.session_state.paper_name
        != uploaded_file.name
    )

    if new_paper:

        with st.spinner(
            "Processing your research paper..."
        ):

            try:

                pages, collection = process_pdf(
                    file_bytes
                )

                st.session_state.pages = pages

                st.session_state.collection = collection

                st.session_state.paper_name = (
                    uploaded_file.name
                )

                st.session_state.summary = None

                st.session_state.history = []

                st.success(
                    "Research paper is ready to explore."
                )

            except Exception as e:

                st.error(
                    f"Unable to process the PDF: {e}"
                )

                st.stop()


# =========================================================
# PAPER INFORMATION
# =========================================================

if st.session_state.collection is not None:

    st.markdown(
        '<div class="section-title">Paper Overview</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Get statistics
    # -----------------------------------------------------

    page_count = len(
        st.session_state.pages
    )

    total_words = sum(
        len(
            p["text"].split()
        )
        for p in st.session_state.pages
    )

    collection_count = (
        st.session_state.collection.count()
    )

    # -----------------------------------------------------
    # Cards
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="card-label">
                    DOCUMENT
                </div>

                <div class="card-value">
                    📄 {st.session_state.paper_name}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="card-label">
                    PAGES
                </div>

                <div class="card-value">
                    {page_count}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="card-label">
                    WORDS
                </div>

                <div class="card-value">
                    {total_words:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # SUMMARY SECTION
    # =====================================================

    st.markdown(
        '<div class="section-title">Research Paper Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Get a structured overview of the research paper.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Generate Summary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating research paper summary..."
        ):

            try:

                summary = generate_summary(
                    st.session_state.pages
                )

                st.session_state.summary = (
                    summary
                )

            except Exception as e:

                st.error(
                    f"Unable to generate summary: {e}"
                )


    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


    # =====================================================
    # QUESTION SECTION
    # =====================================================

    st.markdown(
        '<div class="section-title">Ask About the Paper</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Ask questions and get answers grounded in the uploaded document.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Explanation mode
    # -----------------------------------------------------

    mode_col1, mode_col2 = st.columns(
        [1, 2]
    )

    with mode_col1:

        explanation_mode = st.radio(
            "Explanation level",
            [
                "Simple",
                "Technical"
            ],
            horizontal=True
        )


    # -----------------------------------------------------
    # Question input
    # -----------------------------------------------------

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What methodology is used in this research paper?"
        ),
        label_visibility="visible"
    )


    # =====================================================
    # ASK QUESTION
    # =====================================================

    if st.button(
        "Ask Question",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Searching the paper and preparing your answer..."
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
                        explanation_mode
                    )

                    # -------------------------------------------------
                    # Save question history
                    # -------------------------------------------------

                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "mode": explanation_mode
                        }
                    )


                    # =================================================
                    # ANSWER
                    # =================================================

                    st.markdown(
                        '<div class="section-title">Answer</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        answer
                    )


                    # =================================================
                    # SOURCES
                    # =================================================

                    st.markdown(
                        '<div class="section-title">Sources</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="source-intro">'
                        'Relevant sections retrieved from the original research paper.'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    # -------------------------------------------------
                    # Display actual retrieved chunks
                    # -------------------------------------------------

                    for i in range(
                        len(retrieved_chunks)
                    ):

                        page = metadata[i].get(
                            "page",
                            "Unknown"
                        )

                        chunk = retrieved_chunks[i]

                        distance = distances[i]

                        with st.expander(
                            f"Page {page}  •  Source {i + 1}"
                        ):

                            st.write(
                                chunk
                            )

                            st.caption(
                                f"Source relevance distance: {distance:.4f}"
                            )


                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {e}"
                    )


    # =====================================================
    # QUESTION HISTORY
    # =====================================================

    if st.session_state.history:

        st.markdown(
            '<div class="section-title">Previous Questions</div>',
            unsafe_allow_html=True
        )

        for item in reversed(
            st.session_state.history
        ):

            with st.expander(
                item["question"]
            ):

                st.caption(
                    f"Explanation: {item['mode']}"
                )

                st.write(
                    item["answer"]
                )


# =========================================================
# INITIAL SCREEN
# =========================================================

else:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:55px 20px;
            border:1px solid #e5e7eb;
            border-radius:14px;
            margin-top:20px;
            background:#fafafa;
        ">

            <div style="
                font-size:45px;
                margin-bottom:15px;
            ">
                📄
            </div>

            <div style="
                font-size:22px;
                font-weight:700;
                margin-bottom:8px;
            ">
                Start with a research paper
            </div>

            <div style="
                color:#6b7280;
                font-size:14px;
                max-width:500px;
                margin:auto;
                line-height:1.6;
            ">
                Upload a PDF above to generate a summary,
                ask questions, and explore evidence from
                the original document.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <b>PaperLens</b> · Research Paper Explainer Assistant
        <br>
        AI-powered academic document analysis
    </div>
    """,
    unsafe_allow_html=True
)
