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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
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

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False


# ============================================================
# GEMINI CLIENT
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
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(file_bytes):

    pages = []

    # Convert bytes into a file-like object.
    # This fixes the seek() error.

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
# PROCESS PDF
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
    # Context
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
    # APP NAME
    # --------------------------------------------------------

    st.markdown("### PaperLens")

    st.caption(
        "Research Paper Assistant"
    )

    # --------------------------------------------------------
    # NEW CHAT / NEW PAPER
    # --------------------------------------------------------

    if st.button(
        "＋ New chat",
        use_container_width=True
    ):

        st.session_state.collection = None

        st.session_state.paper_name = None

        st.session_state.pages = []

        st.session_state.summary = None

        st.session_state.history = []

        st.session_state.show_uploader = True

        st.rerun()

    # --------------------------------------------------------
    # RECENTS
    # --------------------------------------------------------

    st.markdown("**Recents**")

    if st.session_state.recents:

        for recent in reversed(
            st.session_state.recents[-8:]
        ):

            st.button(
                recent,
                key=f"recent_{recent}",
                use_container_width=True,
                disabled=True
            )

    else:

        st.caption(
            "No recent papers"
        )

    # --------------------------------------------------------
    # CURRENT PAPER
    # --------------------------------------------------------

    st.markdown("**Current paper**")

    if st.session_state.paper_name:

        st.caption(
            st.session_state.paper_name
        )

        st.caption(
            f"{len(st.session_state.pages)} pages"
        )

    else:

        st.caption(
            "No paper uploaded"
        )

    # --------------------------------------------------------
    # RESPONSE STYLE
    # --------------------------------------------------------

    st.markdown("**Response style**")

    st.session_state.explanation_mode = st.radio(
        "Response style",
        [
            "Simple",
            "Technical"
        ],
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


# ============================================================
# MAIN AREA
# ============================================================

# ============================================================
# NO PAPER
# ============================================================

if not st.session_state.pages:

    st.title(
        "PaperLens"
    )

    st.write(
        "Upload a research paper and start exploring it."
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="visible"
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        with st.spinner(
            "Processing your research paper..."
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
    # TOP BAR
    # --------------------------------------------------------

    st.title(
        st.session_state.paper_name
    )

    st.caption(
        f"{len(st.session_state.pages)} pages"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if st.session_state.summary is None:

        if st.button(
            "Generate summary"
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
    # SHOW SUMMARY
    # --------------------------------------------------------

    if st.session_state.summary:

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                "**Paper summary**"
            )

            st.markdown(
                st.session_state.summary
            )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for item in st.session_state.history:

        # User message
        with st.chat_message(
            "user"
        ):

            st.markdown(
                item["question"]
            )

        # Assistant message
        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                item["answer"]
            )

            # Sources
            if (
                "chunks" in item
                and item["chunks"]
            ):

                st.markdown(
                    "**Sources**"
                )

                for i in range(
                    len(item["chunks"])
                ):

                    page = item[
                        "metadata"
                    ][i].get(
                        "page",
                        "Unknown"
                    )

                    distance = item[
                        "distances"
                    ][i]

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

    question = st.chat_input(
        "Ask anything about this paper..."
    )

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # ----------------------------------------------------
        # AI RESPONSE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

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

                    st.markdown(
                        answer
                    )

                    # ------------------------------------------------
                    # SOURCES
                    # ------------------------------------------------

                    st.markdown(
                        "**Sources**"
                    )

                    for i in range(
                        len(retrieved_chunks)
                    ):

                        page = metadata[i].get(
                            "page",
                            "Unknown"
                        )

                        distance = distances[i]

                        with st.expander(
                            f"Page {page} · Source {i + 1}"
                        ):

                            st.write(
                                retrieved_chunks[i]
                            )

                            st.caption(
                                f"Relevance distance: {distance:.4f}"
                            )

                    # ------------------------------------------------
                    # SAVE HISTORY
                    # ------------------------------------------------

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

                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {e}"
                    )
