import streamlit as st
from pypdf import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import hashlib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

if "recents" not in st.session_state:
    st.session_state.recents = []

if "explanation_mode" not in st.session_state:
    st.session_state.explanation_mode = "Simple"


# =========================================================
# GEMINI CLIENT
# =========================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    client = None


# =========================================================
# MODELS
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_chroma_client():
    return chromadb.Client()


embedding_model = load_embedding_model()
chroma_client = load_chroma_client()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_pages(file_bytes):

    pages = []

    # Convert bytes into a file-like object
    pdf_file = BytesIO(file_bytes)

    reader = PdfReader(pdf_file)

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text:
            page_text = page_text.strip()
        else:
            page_text = ""

        if page_text:
            pages.append({
                "page": page_number,
                "text": page_text
            })

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

        page_chunks = text_splitter.split_text(page_text)

        for chunk in page_chunks:

            chunks.append(chunk)
            page_numbers.append(page_number)

    return chunks, page_numbers


# =========================================================
# COLLECTION NAME
# =========================================================

def create_collection_name(file_bytes):

    file_hash = hashlib.md5(file_bytes).hexdigest()

    return f"research_paper_{file_hash[:10]}"


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(file_bytes):

    pages = extract_pdf_pages(file_bytes)

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    chunks, page_numbers = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    collection_name = create_collection_name(file_bytes)

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    # Add only if collection is empty
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
# SUMMARY GENERATION
# =========================================================

def generate_summary(pages):

    if client is None:
        raise ValueError(
            "Gemini API key is not configured."
        )

    # Limit extremely large papers
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
If something is not clearly available in the paper,
write "Not specified in the paper."

Create a clear academic summary using exactly these sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

Keep the explanation clear and suitable for a college student.

Research Paper:
{full_text}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# =========================================================
# QUESTION ANSWERING / RAG
# =========================================================

def ask_question(
    question,
    collection,
    explanation_mode
):

    if client is None:
        raise ValueError(
            "Gemini API key is not configured."
        )

    # Create question embedding
    question_embedding = embedding_model.encode(
        [question]
    )[0]

    # Retrieve relevant chunks
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
        distances = [0] * len(retrieved_chunks)

    # Combine retrieved context
    context = ""

    for i, chunk in enumerate(retrieved_chunks):

        page = retrieved_metadata[i].get(
            "page",
            "Unknown"
        )

        context += (
            f"\n\n--- Page {page} ---\n"
            f"{chunk}"
        )

    if explanation_mode == "Simple":

        instruction = """
Explain the answer in simple language.
Assume the user is a beginner.
Avoid unnecessary technical jargon.
Use short paragraphs or bullet points when useful.
"""

    else:

        instruction = """
Give a technical and academically detailed answer.
Use appropriate technical terminology.
Explain algorithms, methodology, and concepts accurately.
"""

    prompt = f"""
You are a Research Paper Question Answering Assistant.

Answer the user's question using ONLY the retrieved
research paper context below.

{instruction}

Important rules:
- Do not use outside knowledge.
- Do not invent information.
- Do not assume information that is not present.
- If the answer cannot be found in the retrieved context,
  respond exactly:

"The information is not available in the research paper."

User Question:
{question}

Retrieved Research Paper Context:
{context}
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
# SIDEBAR
# =========================================================

with st.sidebar:

    # ---------- BRAND ----------

    st.title("📚 PaperLens")
    st.caption("Research Paper Assistant")

    # ---------- NEW PAPER ----------

    if st.button(
        "＋ New paper",
        use_container_width=True
    ):

        st.session_state.collection = None
        st.session_state.paper_name = None
        st.session_state.pages = []
        st.session_state.summary = None
        st.session_state.history = []

        st.rerun()

    # ---------- RECENTS ----------

    st.markdown("### Recents")

    if st.session_state.recents:

        # Show latest papers first
        for recent_paper in reversed(
            st.session_state.recents[-8:]
        ):

            if st.button(
                f"📄 {recent_paper}",
                key=f"recent_{recent_paper}",
                use_container_width=True
            ):

                st.info(
                    "Please upload the paper again to reopen it."
                )

    else:

        st.caption("No recent papers yet.")

    # ---------- CURRENT PAPER ----------

    st.markdown("### Current paper")

    if st.session_state.paper_name:

        st.caption(
            f"📄 {st.session_state.paper_name}"
        )

        st.caption(
            f"{len(st.session_state.pages)} pages"
        )

    else:

        st.caption("No paper uploaded")

    # ---------- EXPLANATION MODE ----------

    st.markdown("### Explanation")

    st.session_state.explanation_mode = st.radio(
        "Explanation level",
        ["Simple", "Technical"],
        index=(
            0
            if st.session_state.explanation_mode == "Simple"
            else 1
        ),
        label_visibility="collapsed"
    )

    # ---------- CLEAR SESSION ----------

    st.markdown("")

    if st.button(
        "🗑️ Clear session",
        use_container_width=True
    ):

        st.session_state.collection = None
        st.session_state.paper_name = None
        st.session_state.pages = []
        st.session_state.summary = None
        st.session_state.history = []

        st.rerun()

    # ---------- FOOTER ----------

    st.markdown("---")

    st.caption(
        "PaperLens • Academic AI Assistant"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("📚 PaperLens")

st.caption(
    "Understand research papers with AI-powered summaries and evidence-based answers."
)


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"],
    help="Upload a text-based research paper in PDF format."
)


# =========================================================
# PROCESS UPLOADED PAPER
# =========================================================

if uploaded_file is not None:

    # Only process if this is a new paper
    if (
        st.session_state.paper_name
        != uploaded_file.name
    ):

        file_bytes = uploaded_file.getvalue()

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

                # Add to Recents
                if (
                    uploaded_file.name
                    not in st.session_state.recents
                ):

                    st.session_state.recents.append(
                        uploaded_file.name
                    )

                st.success(
                    "Research paper processed successfully."
                )

            except Exception as e:

                st.error(
                    f"Unable to process PDF: {e}"
                )


# =========================================================
# PAPER OVERVIEW
# =========================================================

if st.session_state.pages:

    st.markdown("## 📄 Paper Overview")

    total_words = sum(
        len(page["text"].split())
        for page in st.session_state.pages
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            len(st.session_state.pages)
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


# =========================================================
# SUMMARY
# =========================================================

if st.session_state.pages:

    st.markdown("## 📝 Research Summary")

    if st.button(
        "Generate Summary",
        use_container_width=False
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

            except Exception as e:

                st.error(
                    f"Unable to generate summary: {e}"
                )

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


# =========================================================
# QUESTION ANSWERING
# =========================================================

if st.session_state.collection is not None:

    st.markdown("---")

    st.markdown("## 💬 Ask About the Paper")

    question = st.text_input(
        "Ask a question",
        placeholder=(
            "Example: What methodology is used in this research paper?"
        )
    )

    if st.button(
        "Ask Question",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the paper and generating an answer..."
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

                    # Store history
                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )

                    # Answer
                    st.markdown("### 💡 Answer")

                    st.write(answer)

                    # Sources
                    st.markdown(
                        "### 📚 Sources"
                    )

                    st.caption(
                        "Relevant passages retrieved from the original research paper."
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

                            st.write(chunk)

                            st.caption(
                                f"Source relevance distance: {distance:.4f}"
                            )

                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {e}"
                    )


# =========================================================
# QUESTION HISTORY
# =========================================================

if st.session_state.history:

    st.markdown("---")

    st.markdown("## 🕘 Question History")

    for item in reversed(
        st.session_state.history
    ):

        with st.expander(
            f"Q: {item['question']}"
        ):

            st.write(
                item["answer"]
            )


# =========================================================
# INITIAL STATE MESSAGE
# =========================================================

if not st.session_state.pages:

    st.info(
        "Upload a research paper PDF to get started."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "PaperLens — Research Paper Explainer Assistant"
)
