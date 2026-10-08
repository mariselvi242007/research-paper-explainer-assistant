# ============================================================
# RESEARCH PAPER EXPLAINER ASSISTANT
# Complete Streamlit Application
# ============================================================

import streamlit as st
import hashlib
import io
import os

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from sentence_transformers import SentenceTransformer

import chromadb

from google import genai

from dotenv import load_dotenv


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research Paper Explainer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# 2. SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

if "collection" not in st.session_state:
    st.session_state.collection = None

if "paper_name" not in st.session_state:
    st.session_state.paper_name = None

if "pages" not in st.session_state:
    st.session_state.pages = 0

if "chunks" not in st.session_state:
    st.session_state.chunks = 0


# ============================================================
# 3. LOAD GEMINI API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


if not api_key:

    st.error(
        "Gemini API key not found."
    )

    st.info(
        "Please create a .env file and add:\n\n"
        "GEMINI_API_KEY=YOUR_API_KEY"
    )

    st.stop()


# ============================================================
# 4. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# 6. CREATE CHROMADB CLIENT
# ============================================================

@st.cache_resource
def get_chroma_client():

    client = chromadb.Client()

    return client


chroma_client = get_chroma_client()


# ============================================================
# 7. EXTRACT PDF TEXT PAGE BY PAGE
# ============================================================

def extract_pdf_text(file_bytes):

    pdf_stream = io.BytesIO(
        file_bytes
    )

    reader = PdfReader(
        pdf_stream
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:

            pages.append(
                {
                    "page": page_number,
                    "text": page_text
                }
            )

    return pages, len(reader.pages)


# ============================================================
# 8. CREATE PAGE-AWARE CHUNKS
# ============================================================

def create_chunks_with_pages(pages):

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

            chunks.append(
                chunk
            )

            page_numbers.append(
                page_data["page"]
            )


    return chunks, page_numbers


# ============================================================
# 9. CREATE UNIQUE COLLECTION NAME
# ============================================================

def create_collection_name(file_bytes):

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return (
        f"research_paper_{file_hash[:10]}"
    )


# ============================================================
# 10. GENERATE PAPER SUMMARY
# ============================================================

def generate_summary(text):

    prompt = f"""
You are a Research Paper Explainer Assistant.

Read the research paper below and create
a clear and organized summary.

Include:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

Use ONLY information available in the
research paper.

Do not invent information.

Use simple and understandable language.

Research Paper:
{text}
"""


    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )


    return response.text


# ============================================================
# 11. ASK QUESTION USING RAG
# ============================================================

def ask_question(
    question,
    collection,
    explanation_mode
):


    # --------------------------------------------------------
    # Create question embedding
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question]
    )


    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    results = collection.query(

        query_embeddings=(
            question_embedding.tolist()
        ),

        n_results=5
    )


    # --------------------------------------------------------
    # Get retrieved chunks
    # --------------------------------------------------------

    retrieved_chunks = (
        results["documents"][0]
    )

    retrieved_metadata = (
        results["metadatas"][0]
    )

    retrieved_ids = (
        results["ids"][0]
    )

    distances = (
        results["distances"][0]
    )


    # --------------------------------------------------------
    # Combine chunks
    # --------------------------------------------------------

    context = "\n\n".join(
        retrieved_chunks
    )


    # --------------------------------------------------------
    # Explanation mode
    # --------------------------------------------------------

    if explanation_mode == "📘 Simple Explanation":

        instruction = """
Explain the answer in very simple language.

Assume the user is a beginner.

Avoid difficult technical terms.

If you use a technical term,
explain it briefly.

Use short paragraphs and bullet points
when useful.
"""


    else:

        instruction = """
Explain the answer using appropriate
technical terminology.

Include important technical details,
algorithms, methods, datasets,
models, and concepts mentioned
in the research paper.
"""


    # --------------------------------------------------------
    # RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are a Research Paper Explainer Assistant.

Use ONLY the research paper context
provided below.

{instruction}

Do not use outside knowledge.

If the answer is not available in the
research paper, say:

"The information is not available
in the research paper."

Research Paper Context:
{context}

Question:
{question}
"""


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = client.models.generate_content(

        model="gemini-3.5-flash",

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
# 12. APPLICATION TITLE
# ============================================================

st.title(
    "📄 Research Paper Explainer Assistant"
)

st.write(
    "Upload a research paper and use AI to "
    "summarize it, ask questions, and understand "
    "the research paper easily."
)

st.divider()


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📄 Research Paper"
    )

    st.subheader(
        "Explainer Assistant"
    )

    st.divider()


    # --------------------------------------------------------
    # Paper information
    # --------------------------------------------------------

    st.write(
        "### 📌 Paper Information"
    )


    if st.session_state.paper_name:

        st.write(
            f"**File:** "
            f"{st.session_state.paper_name}"
        )

        st.write(
            f"**Pages:** "
            f"{st.session_state.pages}"
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state.chunks}"
        )

    else:

        st.info(
            "Upload a PDF to begin."
        )


    st.divider()


    # --------------------------------------------------------
    # Technology information
    # --------------------------------------------------------

    st.write(
        "### 🤖 AI Models"
    )

    st.write(
        "**Embeddings:**"
    )

    st.code(
        "all-MiniLM-L6-v2"
    )


    st.write(
        "**LLM:**"
    )

    st.code(
        "Gemini 3.5 Flash"
    )


    st.write(
        "**Vector Database:**"
    )

    st.code(
        "ChromaDB"
    )


    st.divider()


    # --------------------------------------------------------
    # Reset button
    # --------------------------------------------------------

    if st.button(
        "🗑️ Reset Application"
    ):

        st.session_state.chat_history = []

        st.session_state.summary = None

        st.session_state.current_file_hash = None

        st.session_state.collection = None

        st.session_state.paper_name = None

        st.session_state.pages = 0

        st.session_state.chunks = 0

        st.rerun()


    st.divider()


    st.caption(
        "Research Paper Explainer Assistant"
    )

    st.caption(
        "Python • Streamlit • ChromaDB • Gemini"
    )


# ============================================================
# 14. PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload your research paper",
    type=["pdf"]
)


# ============================================================
# 15. PROCESS UPLOADED PDF
# ============================================================

if uploaded_file:

    file_bytes = uploaded_file.getvalue()


    # --------------------------------------------------------
    # Create file hash
    # --------------------------------------------------------

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()


    # --------------------------------------------------------
    # Detect new paper
    # --------------------------------------------------------

    if (
        file_hash
        != st.session_state.current_file_hash
    ):


        # Clear previous paper information

        st.session_state.chat_history = []

        st.session_state.summary = None


        # Save current paper

        st.session_state.current_file_hash = (
            file_hash
        )

        st.session_state.paper_name = (
            uploaded_file.name
        )


        # ----------------------------------------------------
        # Create unique ChromaDB collection
        # ----------------------------------------------------

        collection_name = (
            create_collection_name(
                file_bytes
            )
        )


        collection = (
            chroma_client.get_or_create_collection(
                name=collection_name
            )
        )


        st.session_state.collection = (
            collection
        )


    else:

        collection = (
            st.session_state.collection
        )


    # ========================================================
    # PROCESS PAPER
    # ========================================================

    with st.spinner(
        "📄 Processing research paper..."
    ):


        # ----------------------------------------------------
        # Extract PDF text
        # ----------------------------------------------------

        pages, number_of_pages = (
            extract_pdf_text(
                file_bytes
            )
        )


        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks, page_numbers = (
            create_chunks_with_pages(
                pages
            )
        )


        # ----------------------------------------------------
        # Create embeddings
        # ----------------------------------------------------

        embeddings = (
            embedding_model.encode(
                chunks
            )
        )


    # Save information

    st.session_state.pages = (
        number_of_pages
    )

    st.session_state.chunks = (
        len(chunks)
    )


    # ========================================================
    # STORE IN CHROMADB
    # ========================================================

    if collection.count() == 0:


        chunk_ids = [

            f"chunk_{i}"

            for i in range(
                len(chunks)
            )
        ]


        metadatas = [

            {
                "page": page_numbers[i]
            }

            for i in range(
                len(chunks)
            )
        ]


        collection.add(

            ids=chunk_ids,

            documents=chunks,

            embeddings=embeddings.tolist(),

            metadatas=metadatas
        )


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        f"✅ Paper processed successfully! "
        f"{len(chunks)} chunks created."
    )


    st.info(
        f"📄 Current paper: "
        f"{uploaded_file.name}"
    )


    # ========================================================
    # PAPER OVERVIEW
    # ========================================================

    st.subheader(
        "📊 Paper Overview"
    )


    full_text = "\n".join(
        page["text"]
        for page in pages
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "📄 Pages",
            number_of_pages
        )


    with col2:

        st.metric(
            "📝 Chunks",
            len(chunks)
        )


    with col3:

        st.metric(
            "🔤 Characters",
            len(full_text)
        )


    with col4:

        st.metric(
            "🔎 Retrieved",
            "Top 5"
        )


    st.divider()


    # ========================================================
    # PAPER SUMMARY
    # ========================================================

    st.subheader(
        "📋 Paper Summary"
    )


    if st.button(
        "📝 Generate Summary"
    ):

        with st.spinner(
            "Generating paper summary..."
        ):

            st.session_state.summary = (
                generate_summary(
                    full_text
                )
            )


    if st.session_state.summary:

        st.write(
            st.session_state.summary
        )


        st.download_button(

            label="📥 Download Summary",

            data=st.session_state.summary,

            file_name=(
                "research_paper_summary.txt"
            ),

            mime="text/plain"
        )


    st.divider()


    # ========================================================
    # EXPLANATION MODE
    # ========================================================

    st.subheader(
        "🎓 Explanation Mode"
    )


    explanation_mode = st.radio(

        "Choose how you want the answer:",

        [
            "📘 Simple Explanation",
            "🔬 Technical Explanation"
        ]
    )


    # ========================================================
    # ASK QUESTION
    # ========================================================

    st.subheader(
        "💬 Ask a Question"
    )


    question = st.text_input(

        "Enter your question:",

        placeholder=(
            "Example: What dataset was used?"
        )
    )


    if st.button(
        "🔍 Ask Question"
    ):


        if question.strip():


            with st.spinner(
                "🤖 Finding the answer..."
            ):


                (
                    answer,
                    retrieved_chunks,
                    retrieved_metadata,
                    retrieved_ids,
                    distances
                ) = ask_question(

                    question,

                    collection,

                    explanation_mode
                )


            # ------------------------------------------------
            # Save chat history
            # ------------------------------------------------

            st.session_state.chat_history.append(

                {
                    "question": question,

                    "answer": answer
                }
            )


            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            st.subheader(
                "🤖 Answer"
            )

            st.write(
                answer
            )


            # =================================================
            # SOURCE REFERENCES
            # =================================================

            st.subheader(
                "📚 Source References"
            )


            st.caption(
                f"{len(retrieved_chunks)} "
                "relevant sections retrieved"
            )


            for i, chunk in enumerate(
                retrieved_chunks
            ):


                page_number = (
                    retrieved_metadata[i]["page"]
                )


                with st.expander(

                    f"📄 Source {i + 1} "
                    f"— Page {page_number}"

                ):


                    st.write(
                        chunk
                    )


                    st.caption(

                        f"Chunk ID: "
                        f"{retrieved_ids[i]}"

                    )


                    st.caption(

                        f"ChromaDB distance: "
                        f"{distances[i]:.4f}"

                    )


        else:

            st.warning(
                "Please enter a question."
            )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if st.session_state.chat_history:

        st.divider()


        st.subheader(
            "💬 Chat History"
        )


        for chat in (
            st.session_state.chat_history
        ):


            st.markdown(
                f"**👤 You:** "
                f"{chat['question']}"
            )


            st.markdown(
                f"**🤖 Assistant:** "
                f"{chat['answer']}"
            )


            st.divider()


else:

    # ========================================================
    # NO PDF UPLOADED
    # ========================================================

    st.info(
        "👆 Upload a research paper PDF "
        "to get started."
    )
