import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import hashlib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# TITLE
# =========================================================

st.title("📚 PaperLens")

st.caption(
    "Research Paper Explainer Assistant"
)

st.write(
    "Upload a research paper, generate a structured summary, "
    "and ask questions about the document."
)

st.divider()


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
# SIDEBAR
# =========================================================
# =========================================================
# PROFESSIONAL COMPACT SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 PaperLens")

    st.caption("Research Paper Assistant")

    st.divider()

    # -----------------------------------------------------
    # CURRENT DOCUMENT
    # -----------------------------------------------------

    st.subheader("Current Document")

    if st.session_state.paper_name:

        st.write(
            f"📄 {st.session_state.paper_name}"
        )

        st.caption(
            f"{len(st.session_state.pages)} pages"
        )

    else:

        st.caption(
            "No paper uploaded yet."
        )


    st.divider()


    # -----------------------------------------------------
    # EXPLANATION PREFERENCE
    # -----------------------------------------------------

    st.subheader("Explanation Level")

    sidebar_mode = st.radio(
        "Choose how answers should be explained",
        ["Simple", "Technical"],
        index=0,
        label_visibility="collapsed"
    )


    st.divider()


    # -----------------------------------------------------
    # QUICK ACTIONS
    # -----------------------------------------------------

    st.subheader("Quick Actions")

    if st.session_state.collection is not None:

        if st.button(
            "📝 Generate Summary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating summary..."
            ):

                try:

                    st.session_state.summary = (
                        generate_summary(
                            st.session_state.pages
                        )
                    )

                    st.success(
                        "Summary generated."
                    )

                except Exception as e:

                    st.error(
                        f"Unable to generate summary: {e}"
                    )

    else:

        st.button(
            "📝 Generate Summary",
            disabled=True,
            use_container_width=True
        )


    if st.button(
        "🗑️ Clear Session",
        use_container_width=True
    ):

        st.session_state.collection = None
        st.session_state.paper_name = None
        st.session_state.pages = []
        st.session_state.summary = None
        st.session_state.history = []

        st.rerun()


    st.divider()


    # -----------------------------------------------------
    # HOW TO USE
    # -----------------------------------------------------

    st.subheader("How to Use")

    st.caption(
        "1. Upload a PDF"
    )

    st.caption(
        "2. Generate a summary"
    )

    st.caption(
        "3. Ask questions"
    )

    st.caption(
        "4. Check the sources"
    )


    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.divider()

    st.caption(
        "PaperLens • Academic AI Assistant"
    )

# =========================================================
# PDF UPLOAD
# =========================================================

st.header("📄 Upload Research Paper")

st.write(
    "Select a PDF research paper to begin."
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# =========================================================
# EXTRACT PDF PAGES
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
# CREATE CHUNKS
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
# UNIQUE COLLECTION
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

Use exactly these sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

Explain each section clearly and accurately.

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
# ASK QUESTION
# =========================================================

def ask_question(
    question,
    collection,
    explanation_mode
):

    question_embedding = embedding_model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=5
    )

    retrieved_chunks = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    retrieved_ids = results["ids"][0]

    distances = results["distances"][0]

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

    if explanation_mode == "Simple":

        mode_instruction = """
Explain the answer in simple English.

Assume the user is a beginner.

Use short sentences.

Explain difficult technical terms briefly.
"""

    else:

        mode_instruction = """
Explain the answer at a technical level.

Include algorithms, models, methodology,
technical concepts, and reasoning when
available in the paper.
"""

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
# PROCESS UPLOADED PAPER
# =========================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

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
                    "Research paper is ready."
                )

            except Exception as e:

                st.error(
                    f"Unable to process PDF: {e}"
                )

                st.stop()


# =========================================================
# MAIN APPLICATION
# =========================================================

if st.session_state.collection is not None:

    # =====================================================
    # PAPER OVERVIEW
    # =====================================================

    st.header("📋 Paper Overview")

    page_count = len(
        st.session_state.pages
    )

    total_words = sum(
        len(
            page["text"].split()
        )
        for page in st.session_state.pages
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            page_count
        )

    with col2:

        st.metric(
            "Words",
            f"{total_words:,}"
        )

    with col3:

        st.metric(
            "Status",
            "Ready"
        )

    st.write(
        f"**Document:** {st.session_state.paper_name}"
    )

    st.divider()


    # =====================================================
    # SUMMARY
    # =====================================================

    st.header("📝 Research Paper Summary")

    st.write(
        "Generate a structured summary of the paper."
    )

    if st.button(
        "Generate Summary",
        type="primary",
        use_container_width=True
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
                    f"Unable to generate summary: {e}"
                )

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )

    st.divider()


    # =====================================================
    # ASK QUESTIONS
    # =====================================================

    st.header("💬 Ask Questions")

    st.write(
        "Ask anything about the uploaded research paper."
    )

    explanation_mode = st.radio(
        "Explanation level",
        [
            "Simple",
            "Technical"
        ],
        horizontal=True
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What methodology is used in this research paper?"
        )
    )

    if st.button(
        "🔍 Ask Question",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Finding relevant information..."
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

                    st.subheader("Answer")

                    st.markdown(
                        answer
                    )


                    # =====================================
                    # SOURCES
                    # =====================================

                    st.subheader("📚 Sources")

                    st.caption(
                        "Relevant passages retrieved from the "
                        "original research paper."
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
                                f"Source relevance distance: "
                                f"{distance:.4f}"
                            )


                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {e}"
                    )


    # =====================================================
    # HISTORY
    # =====================================================

    if st.session_state.history:

        st.divider()

        st.header("🕘 Previous Questions")

        for item in reversed(
            st.session_state.history
        ):

            with st.expander(
                item["question"]
            ):

                st.caption(
                    f"Explanation level: {item['mode']}"
                )

                st.write(
                    item["answer"]
                )


# =========================================================
# INITIAL SCREEN
# =========================================================

else:

    st.info(
        "Upload a PDF research paper above to get started."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PaperLens • Research Paper Explainer Assistant"
)
