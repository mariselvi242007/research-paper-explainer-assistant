import streamlit as st
from pypdf import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
import hashlib


# ============================================================
# PAGE CONFIGURATION
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

    # Convert bytes to file-like object.
    # Fixes:
    # 'bytes' object has no attribute 'seek'

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
# SUMMARY GENERATION
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
# QUESTION ANSWERING
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

    # Question embedding

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

        distances = [
            0
            for _ in retrieved_chunks
        ]

    # Build context

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

    # Explanation mode

    if explanation_mode == "Simple":

        instruction = """
Explain the answer in simple language.

Assume the user is a beginner.

Avoid unnecessary technical jargon.

Use clear paragraphs and bullet points
when useful.
"""

    else:

        instruction = """
Give a technical and academically detailed answer.

Use appropriate technical terminology.

Explain algorithms, methodology and concepts
accurately.
"""

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
- Base the answer on the research paper.
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

    st.title("PaperLens")

    st.caption(
        "Research Paper Assistant"
    )

    st.divider()

    # New chat

    if st.button(
        "＋ New chat",
        use_container_width=True
    ):

        st.session_state.collection = None
        st.session_state.paper_name = None
        st.session_state.pages = []
        st.session_state.summary = None
        st.session_state.history = []

        st.rerun()

    # Recents

    st.markdown("**Recents**")

    if st.session_state.recents:

        for recent in reversed(
            st.session_state.recents[-8:]
        ):

            st.caption(
                f"• {recent}"
            )

    else:

        st.caption(
            "No recent papers"
        )

    st.divider()

    # Current paper

    st.markdown("**Current paper**")

    if st.session_state.paper_name:

        st.write(
            st.session_state.paper_name
        )

        st.caption(
            f"{len(st.session_state.pages)} pages"
        )

    else:

        st.caption(
            "No paper uploaded"
        )

    # Response style

    st.divider()

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

    st.divider()

    # Clear

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
# MAIN CONTENT
# ============================================================

if not st.session_state.pages:

    # --------------------------------------------------------
    # WELCOME
    # --------------------------------------------------------

    st.title(
        "PaperLens"
    )

    st.subheader(
        "Understand your research paper"
    )

    st.write(
        "Upload a research paper to generate a summary, "
        "ask questions, and explore the supporting passages."
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload research paper",
        type=["pdf"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        with st.spinner(
            "Processing research paper..."
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
# PAPER INTERFACE
# ============================================================

else:

    # --------------------------------------------------------
    # TOP
    # --------------------------------------------------------

    st.title(
        st.session_state.paper_name
    )

    total_words = sum(
        len(page["text"].split())
        for page in st.session_state.pages
    )

    st.caption(
        f"{len(st.session_state.pages)} pages  ·  "
        f"{total_words:,} words"
    )

    st.divider()

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [1, 1, 3]
    )

    with col1:

        if st.button(
            "Generate summary",
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

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Unable to generate summary: {e}"
                    )

    with col2:

        st.write("")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if st.session_state.summary:

        st.subheader(
            "Paper Summary"
        )

        st.markdown(
            st.session_state.summary
        )

        st.divider()

    # --------------------------------------------------------
    # CONVERSATION
    # --------------------------------------------------------

    if st.session_state.history:

        st.subheader(
            "Conversation"
        )

        for item in st.session_state.history:

            # ------------------------------------------------
            # QUESTION / ANSWER LAYOUT
            # ------------------------------------------------

            question_col, answer_col = st.columns(
                [1, 2]
            )

            # Question - LEFT

            with question_col:

                st.markdown(
                    "**You**"
                )

                st.info(
                    item["question"]
                )

            # Answer - RIGHT

            with answer_col:

                st.markdown(
                    "**PaperLens**"
                )

                st.write(
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
                                f"Relevance distance: "
                                f"{distance:.4f}"
                            )

            st.divider()

    else:

        # ----------------------------------------------------
        # EMPTY CONVERSATION
        # ----------------------------------------------------

        st.subheader(
            "Ask questions about your paper"
        )

        st.write(
            "You can ask about the methodology, "
            "dataset, algorithms, results, limitations, "
            "or any other information contained in the paper."
        )

        st.write("")

        example_col1, example_col2 = st.columns(2)

        with example_col1:

            st.info(
                "What methodology is used in this paper?"
            )

            st.info(
                "What dataset was used?"
            )

        with example_col2:

            st.info(
                "What are the main results?"
            )

            st.info(
                "What are the limitations?"
            )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.collection is not None:

    question = st.chat_input(
        "Ask anything about this research paper..."
    )

    if question:

        with st.spinner(
            "Finding the relevant information..."
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
