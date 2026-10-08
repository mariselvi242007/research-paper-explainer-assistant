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
    page_title="Research Paper Explainer",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #777777;
        font-size: 14px;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📚 Research Paper Explainer Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a research paper and ask questions using AI-powered Retrieval-Augmented Generation.'
    '</div>',
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
        "Please add it in Streamlit Secrets."
    )
    st.stop()

client = genai.Client(api_key=api_key)


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
# PDF PAGE EXTRACTION
# =========================================================

def extract_pdf_pages(file_bytes):

    pages = []

    reader = PdfReader(file_bytes)

    for page_number, page in enumerate(reader.pages, start=1):

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
# CREATE PAGE-AWARE CHUNKS
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
            page_numbers.append(page_number)

    return chunks, page_numbers


# =========================================================
# CREATE UNIQUE CHROMA COLLECTION NAME
# =========================================================

def create_collection_name(file_bytes):

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return f"research_paper_{file_hash[:10]}"


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

    # Avoid duplicate insertion when the same paper
    # is processed again during the Streamlit session.
    try:

        existing_count = collection.count()

        if existing_count == 0:

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

    except Exception:

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
# GENERATE PAPER SUMMARY
# =========================================================

def generate_summary(pages):

    full_text = "\n\n".join(
        [
            f"Page {p['page']}:\n{p['text']}"
            for p in pages
        ]
    )

    # Limit very large papers for this demo.
    full_text = full_text[:100000]

    prompt = f"""
You are a Research Paper Explainer Assistant.

Summarize ONLY the research paper provided below.

Do not use outside knowledge.

Start the answer directly with:

1. Research Objective

Do not write an introduction such as:
"This summary is based on..."
Do not mention that you are an AI.

Use exactly these sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

Explain each section clearly using simple but technically correct language.

Research Paper:
{full_text}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# =========================================================
# ASK QUESTION USING RAG
# =========================================================

def ask_question(
    question,
    collection,
    explanation_mode
):

    # -----------------------------------------------------
    # Convert question into embedding
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
    # Create context
    # -----------------------------------------------------

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):

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
Explain the answer in very simple English.

Assume the user is a beginner.

Use short sentences and easy technical explanations.
"""

    else:

        mode_instruction = """
Explain the answer at a technical level.

Include relevant algorithms, methods,
models, technical concepts, and reasoning
when they are available in the paper.
"""

    # -----------------------------------------------------
    # Gemini prompt
    # -----------------------------------------------------

    prompt = f"""
You are a Research Paper Explainer Assistant.

Answer the user's question using ONLY the
research paper context provided below.

Do not use outside knowledge.

{mode_instruction}

If the information is not available in the
research paper, say exactly:

"The information is not available in the research paper."

Do not invent information.

Research Paper Context:
{context}

User Question:
{question}
"""

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Assistant")

    st.write(
        "Upload a research paper and use the "
        "assistant to understand it."
    )

    st.divider()

    st.subheader("Features")

    st.write("📄 PDF Upload")
    st.write("📝 Automatic Summary")
    st.write("🔎 Question Answering")
    st.write("🧠 Simple / Technical Mode")
    st.write("📚 Page-Based Sources")
    st.write("🔗 RAG-based Retrieval")

    st.divider()

    st.caption(
        "Embeddings, vector database and retrieval "
        "work in the backend."
    )


# =========================================================
# PDF UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📄 Upload Research Paper</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# =========================================================
# PROCESS UPLOADED PDF
# =========================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    # Detect whether this is a new paper
    new_paper = (
        st.session_state.paper_name
        != uploaded_file.name
    )

    if new_paper:

        with st.spinner(
            "Processing research paper..."
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
                    "Research paper processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )

                st.stop()


# =========================================================
# PAPER INFORMATION
# =========================================================

if st.session_state.collection is not None:

    st.markdown(
        '<div class="section-title">📋 Paper Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            len(st.session_state.pages)
        )

    with col2:

        st.metric(
            "Paper",
            st.session_state.paper_name
        )

    with col3:

        st.metric(
            "Knowledge Base",
            "Ready"
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">📝 Research Paper Summary</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Generate Summary",
        use_container_width=False
    ):

        with st.spinner(
            "Generating summary..."
        ):

            try:

                summary = generate_summary(
                    st.session_state.pages
                )

                st.session_state.summary = summary

            except Exception as e:

                st.error(
                    f"Error generating summary: {e}"
                )


    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    st.markdown(
        '<div class="section-title">🤖 Ask Questions</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Ask any question about the uploaded research paper."
    )

    explanation_mode = st.radio(
        "Explanation Mode",
        [
            "Simple",
            "Technical"
        ],
        horizontal=True
    )

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What methodology is used in this research paper?"
    )


    # =====================================================
    # ASK BUTTON
    # =====================================================

    if st.button(
        "🔍 Ask Question",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the research paper..."
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

                    # Save history
                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "mode": explanation_mode
                        }
                    )

                    # =====================================
                    # ANSWER
                    # =====================================

                    st.markdown(
                        '<div class="section-title">'
                        '🤖 Answer'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        answer
                    )


                    # =====================================
                    # SOURCES
                    # =====================================

                    st.markdown(
                        '<div class="section-title">'
                        '📚 Sources'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        "These are the relevant sections "
                        "retrieved from the research paper."
                    )

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
                            f"📄 Page {page} — Source {i + 1}"
                        ):

                            st.write(
                                chunk
                            )

                            st.caption(
                                f"Retrieval distance: {distance:.4f}"
                            )


                except Exception as e:

                    st.error(
                        f"Error answering question: {e}"
                    )


    # =====================================================
    # QUESTION HISTORY
    # =====================================================

    if st.session_state.history:

        st.markdown(
            '<div class="section-title">'
            '🕘 Previous Questions'
            '</div>',
            unsafe_allow_html=True
        )

        for i, item in enumerate(
            reversed(st.session_state.history)
        ):

            with st.expander(
                f"Question: {item['question']}"
            ):

                st.caption(
                    f"Mode: {item['mode']}"
                )

                st.write(
                    item["answer"]
                )


# =========================================================
# NO PDF MESSAGE
# =========================================================

else:

    st.info(
        "👆 Upload a research paper PDF to get started."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Research Paper Explainer Assistant |
        Powered by RAG, ChromaDB, Sentence Transformers and Gemini
    </div>
    """,
    unsafe_allow_html=True
)
