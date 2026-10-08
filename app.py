import streamlit as st
import hashlib
import io

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research Paper Explainer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS - FRONTEND DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Answer box */
    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 10px;
        line-height: 1.7;
    }

    /* Source box */
    .source-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    /* Small information text */
    .info-text {
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📄 Research Paper Explainer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a research paper and understand it easily with AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GEMINI API
# ============================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]

except Exception:
    st.error(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()


client = genai.Client(api_key=api_key)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_chroma_client():

    return chromadb.Client()


chroma_client = load_chroma_client()


# ============================================================
# SESSION STATE
# ============================================================

if "paper_processed" not in st.session_state:
    st.session_state.paper_processed = False

if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "collection" not in st.session_state:
    st.session_state.collection = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(file_bytes):

    pdf_stream = io.BytesIO(file_bytes)

    reader = PdfReader(pdf_stream)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text and page_text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": page_text
                }
            )

    return pages, len(reader.pages)


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    page_numbers = []

    for page_data in pages:

        page_chunks = text_splitter.split_text(
            page_data["text"]
        )

        for chunk in page_chunks:

            chunks.append(chunk)

            page_numbers.append(
                page_data["page"]
            )

    return chunks, page_numbers


# ============================================================
# CREATE UNIQUE COLLECTION NAME
# ============================================================

def create_collection_name(file_bytes):

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return "paper_" + file_hash[:12]


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(file_bytes):

    pages, total_pages = extract_pdf_pages(
        file_bytes
    )

    chunks, page_numbers = create_chunks(
        pages
    )

    if not chunks:

        raise ValueError(
            "No readable text was found in the PDF."
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

    # Add only if the collection is empty
    if collection.count() == 0:

        ids = [
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
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    return (
        collection,
        total_pages,
        len(chunks)
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(pages):

    paper_text = "\n\n".join(
        page["text"]
        for page in pages
    )

    # Limit text to avoid unnecessarily huge requests
    paper_text = paper_text[:100000]

    prompt = f"""
You are a Research Paper Explainer Assistant.

Read the research paper below.

Create a clear and well-organized summary.

Include these sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

Rules:

- Use ONLY information from the research paper.
- Do not invent information.
- Do not use outside knowledge.
- Explain technical terms briefly.
- Use simple language.
- Use bullet points where useful.

Research Paper:

{paper_text}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    question,
    collection,
    explanation_mode
):

    # --------------------------------------------------------
    # Convert question into embedding
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )

    # --------------------------------------------------------
    # Search relevant paper chunks
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=5
    )

    retrieved_chunks = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    retrieved_ids = results["ids"][0]

    distances = results["distances"][0]

    context = "\n\n".join(
        retrieved_chunks
    )

    # --------------------------------------------------------
    # Explanation mode
    # --------------------------------------------------------

    if explanation_mode == "Simple":

        instruction = """
Explain the answer in very simple language.

Assume the reader is a beginner.

Avoid difficult technical words.

If a technical term is necessary,
briefly explain its meaning.

Use short paragraphs and bullet points.
"""

    else:

        instruction = """
Give a detailed technical explanation.

Use the technical terminology from
the research paper.

Mention important algorithms,
methods, models, datasets and
technical concepts when relevant.
"""

    # --------------------------------------------------------
    # Gemini prompt
    # --------------------------------------------------------

    prompt = f"""
You are a Research Paper Explainer Assistant.

Answer the user's question using ONLY
the research paper context below.

{instruction}

IMPORTANT:

- Do not use outside knowledge.
- Do not invent information.
- If the answer is not available in the
  provided research paper context, say:

"The information is not available
in the research paper."

Research Paper Context:

{context}

User Question:

{question}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return (
        response.text,
        retrieved_metadata,
        retrieved_ids,
        distances
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📄 Paper Assistant")

    st.markdown(
        """
        Upload a research paper and ask questions
        about its content.
        """
    )

    st.divider()

    st.markdown("### ✨ Features")

    st.write("📤 Upload research paper")

    st.write("📝 Generate summary")

    st.write("💬 Ask questions")

    st.write("📘 Simple explanation")

    st.write("🔬 Technical explanation")

    st.write("📚 Page-based sources")

    st.divider()

    st.caption(
        "Research Paper Explainer Assistant"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📤 Upload Your Research Paper'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ============================================================
# PROCESS UPLOADED PAPER
# ============================================================

if uploaded_file is not None:

    current_file_bytes = uploaded_file.getvalue()

    current_file_hash = hashlib.md5(
        current_file_bytes
    ).hexdigest()

    previous_file_hash = st.session_state.get(
        "file_hash"
    )

    # Process only when a new PDF is uploaded
    if current_file_hash != previous_file_hash:

        with st.spinner(
            "Processing your research paper..."
        ):

            try:

                (
                    collection,
                    page_count,
                    chunk_count
                ) = process_pdf(
                    current_file_bytes
                )

                st.session_state.collection = collection

                st.session_state.paper_name = (
                    uploaded_file.name
                )

                st.session_state.page_count = (
                    page_count
                )

                st.session_state.chunk_count = (
                    chunk_count
                )

                st.session_state.file_hash = (
                    current_file_hash
                )

                st.session_state.paper_processed = True

                st.session_state.summary = None

                st.session_state.chat_history = []

                st.success(
                    "Research paper processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Unable to process the PDF: {e}"
                )

    # ========================================================
    # PAPER INFORMATION
    # ========================================================

    if st.session_state.paper_processed:

        st.markdown(
            '<div class="section-title">'
            '📑 Paper Information'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Paper",
                st.session_state.paper_name
            )

        with col2:

            st.metric(
                "Pages",
                st.session_state.page_count
            )

        with col3:

            st.metric(
                "Ready",
                "Yes"
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📝 Paper Summary'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Generate Summary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating paper summary..."
            ):

                try:

                    pages, _ = extract_pdf_pages(
                        current_file_bytes
                    )

                    st.session_state.summary = (
                        generate_summary(pages)
                    )

                except Exception as e:

                    st.error(
                        f"Unable to generate summary: {e}"
                    )


        if st.session_state.summary:

            st.markdown(
                '<div class="answer-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                st.session_state.summary
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # QUESTION ANSWERING
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '💬 Ask Questions About Your Paper'
            '</div>',
            unsafe_allow_html=True
        )

        explanation_mode = st.radio(
            "Explanation type",
            [
                "Simple",
                "Technical"
            ],
            horizontal=True
        )

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What methodology is used "
                "in this research paper?"
            )
        )

        ask_button = st.button(
            "🤖 Ask Question",
            use_container_width=True
        )


        # ====================================================
        # ANSWER
        # ====================================================

        if ask_button:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Finding the answer..."
                ):

                    try:

                        (
                            answer,
                            metadata,
                            ids,
                            distances
                        ) = ask_question(
                            question,
                            st.session_state.collection,
                            explanation_mode
                        )

                        st.session_state.chat_history.append(
                            {
                                "question": question,
                                "answer": answer
                            }
                        )

                        st.markdown(
                            '<div class="section-title">'
                            '🤖 Answer'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            '<div class="answer-box">',
                            unsafe_allow_html=True
                        )

                        st.markdown(answer)

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )


                        # ====================================
                        # SOURCES
                        # ====================================

                        st.markdown(
                            '<div class="section-title">'
                            '📚 Sources'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        unique_pages = []

                        for metadata_item in metadata:

                            page = metadata_item.get(
                                "page"
                            )

                            if page not in unique_pages:

                                unique_pages.append(
                                    page
                                )

                        for page in sorted(
                            unique_pages
                        ):

                            st.markdown(
                                f"""
                                <div class="source-box">
                                📄 <b>Page {page}</b>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    except Exception as e:

                        st.error(
                            f"Unable to answer the question: {e}"
                        )


        # ====================================================
        # PREVIOUS QUESTIONS
        # ====================================================

        if st.session_state.chat_history:

            st.markdown(
                '<div class="section-title">'
                '🕘 Previous Questions'
                '</div>',
                unsafe_allow_html=True
            )

            for item in reversed(
                st.session_state.chat_history
            ):

                with st.expander(
                    item["question"]
                ):

                    st.write(
                        item["answer"]
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📄 Research Paper Explainer Assistant | "
    "AI-powered research paper understanding"
)
