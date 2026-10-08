import streamlit as st
import hashlib
from io import BytesIO

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


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

defaults = {
    "collection": None,
    "paper_name": None,
    "pages": [],
    "summary": None,

    # Current chat
    "current_chat_id": None,

    # All chats
    "chats": {},

    # Recent chat order
    "recents": [],

    # Pinned chats
    "pinned": [],

    # Explanation mode
    "explanation_mode": "Simple",

    # Upload key
    "upload_key": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GEMINI CLIENT
# ============================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]

    client = genai.Client(
        api_key=api_key
    )

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

    pdf_file = BytesIO(file_bytes)

    reader = PdfReader(
        pdf_file
    )

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
            page_numbers.append(page_number)

    return chunks, page_numbers


# ============================================================
# UNIQUE CHROMA COLLECTION
# ============================================================

def create_collection_name(file_bytes):

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    return f"research_paper_{file_hash[:10]}"


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
# CREATE CHAT
# ============================================================

def create_chat(
    paper_name,
    collection,
    pages
):

    chat_id = hashlib.md5(
        f"{paper_name}{len(st.session_state.chats)}".encode()
    ).hexdigest()[:10]

    st.session_state.chats[chat_id] = {

        "id": chat_id,

        "title": paper_name,

        "paper_name": paper_name,

        "collection": collection,

        "pages": pages,

        "summary": None,

        "history": []

    }

    st.session_state.recents.insert(
        0,
        chat_id
    )

    st.session_state.current_chat_id = chat_id

    return chat_id


# ============================================================
# GET CURRENT CHAT
# ============================================================

def get_current_chat():

    chat_id = st.session_state.current_chat_id

    if not chat_id:
        return None

    return st.session_state.chats.get(
        chat_id
    )


# ============================================================
# OPEN CHAT
# ============================================================

def open_chat(chat_id):

    if chat_id not in st.session_state.chats:
        return

    st.session_state.current_chat_id = chat_id

    if chat_id in st.session_state.recents:

        st.session_state.recents.remove(
            chat_id
        )

    st.session_state.recents.insert(
        0,
        chat_id
    )


# ============================================================
# NEW CHAT
# ============================================================

def new_chat():

    st.session_state.current_chat_id = None

    st.session_state.collection = None

    st.session_state.paper_name = None

    st.session_state.pages = []

    st.session_state.summary = None

    st.session_state.upload_key += 1


# ============================================================
# DELETE CHAT
# ============================================================

def delete_chat(chat_id):

    if chat_id in st.session_state.chats:

        del st.session_state.chats[
            chat_id
        ]

    if chat_id in st.session_state.recents:

        st.session_state.recents.remove(
            chat_id
        )

    if chat_id in st.session_state.pinned:

        st.session_state.pinned.remove(
            chat_id
        )

    if st.session_state.current_chat_id == chat_id:

        new_chat()


# ============================================================
# PIN / UNPIN
# ============================================================

def toggle_pin(chat_id):

    if chat_id in st.session_state.pinned:

        st.session_state.pinned.remove(
            chat_id
        )

    else:

        st.session_state.pinned.insert(
            0,
            chat_id
        )


# ============================================================
# SUMMARY GENERATION
# ============================================================

def generate_summary(pages):

    paper_text = "\n\n".join(
        [
            f"PAGE {p['page']}\n{p['text']}"
            for p in pages
        ]
    )

    prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the research paper provided below.

Do not use outside knowledge.
Do not invent information.

Create a structured summary using EXACTLY these sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Dataset
5. Algorithms / Technologies Used
6. Main Results
7. Limitations
8. Conclusion

If information is not available in the paper, write:

Not specified in the paper.

Research Paper:
{paper_text}
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

    question_embedding = embedding_model.encode(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=5
    )

    retrieved_chunks = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    retrieved_ids = results["ids"][0]

    distances = results["distances"][0]

    context = "\n\n".join(
        [
            f"""
SOURCE {i + 1}
PAGE: {retrieved_metadata[i].get('page', 'Unknown')}

TEXT:
{retrieved_chunks[i]}
"""
            for i in range(len(retrieved_chunks))
        ]
    )

    if explanation_mode == "Simple":

        style_instruction = """
Explain the answer in simple English.
Use beginner-friendly language.
Avoid unnecessary technical terminology.
"""

    else:

        style_instruction = """
Give a technical and detailed explanation.
Use appropriate research and computer science terminology.
"""

    prompt = f"""
You are PaperLens, a research paper question-answering assistant.

Answer the user's question using ONLY the retrieved content from the research paper.

Do NOT use outside knowledge.

Do NOT invent information.

{style_instruction}

If the answer cannot be found in the retrieved paper content, say exactly:

The information is not available in the research paper.

User Question:
{question}

Retrieved Paper Content:
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
    # BRAND
    # --------------------------------------------------------

    st.markdown(
        "## PaperLens"
    )

    st.caption(
        "Research Paper Assistant"
    )

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋  New chat",
        use_container_width=True
    ):

        new_chat()

        st.rerun()

    # --------------------------------------------------------
    # RESPONSE STYLE
    # --------------------------------------------------------

    st.markdown(
        "**Response style**"
    )

    explanation_mode = st.radio(
        "Response style",
        [
            "Simple",
            "Technical"
        ],
        index=(
            0
            if st.session_state.explanation_mode == "Simple"
            else 1
        ),
        horizontal=True,
        label_visibility="collapsed"
    )

    st.session_state.explanation_mode = (
        explanation_mode
    )

    st.divider()

    # --------------------------------------------------------
    # PINNED
    # --------------------------------------------------------

    if st.session_state.pinned:

        st.markdown(
            "**Pinned**"
        )

        for chat_id in list(
            st.session_state.pinned
        ):

            if chat_id not in st.session_state.chats:
                continue

            chat = st.session_state.chats[
                chat_id
            ]

            is_current = (
                chat_id
                ==
                st.session_state.current_chat_id
            )

            label = (
                "● "
                if is_current
                else ""
            ) + chat["title"]

            if st.button(
                label,
                key=f"pinned_{chat_id}",
                use_container_width=True
            ):

                open_chat(chat_id)

                st.rerun()

    # --------------------------------------------------------
    # RECENTS
    # --------------------------------------------------------

    st.markdown(
        "**Recents**"
    )

    recent_chat_ids = [
        chat_id
        for chat_id in st.session_state.recents
        if chat_id in st.session_state.chats
        and chat_id not in st.session_state.pinned
    ]

    if recent_chat_ids:

        for chat_id in recent_chat_ids:

            chat = st.session_state.chats[
                chat_id
            ]

            is_current = (
                chat_id
                ==
                st.session_state.current_chat_id
            )

            label = (
                "● "
                if is_current
                else ""
            ) + chat["title"]

            if st.button(
                label,
                key=f"recent_{chat_id}",
                use_container_width=True
            ):

                open_chat(chat_id)

                st.rerun()

    else:

        st.caption(
            "No recent chats"
        )

    # --------------------------------------------------------
    # CURRENT CHAT ACTIONS
    # --------------------------------------------------------

    current_chat = get_current_chat()

    if current_chat:

        st.divider()

        st.markdown(
            "**Chat options**"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Pin"
                if current_chat["id"]
                not in st.session_state.pinned
                else "Unpin",
                use_container_width=True
            ):

                toggle_pin(
                    current_chat["id"]
                )

                st.rerun()

        with col2:

            if st.button(
                "Clear",
                use_container_width=True
            ):

                current_chat["history"] = []

                current_chat["summary"] = None

                st.rerun()

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "PaperLens • RAG Research Assistant"
    )


# ============================================================
# MAIN AREA
# ============================================================

current_chat = get_current_chat()


# ============================================================
# NO PAPER / NEW CHAT SCREEN
# ============================================================

if current_chat is None:

    st.markdown(
        "# PaperLens"
    )

    st.markdown(
        "Upload a research paper to start a conversation."
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload research paper",
        type=["pdf"],
        key=f"pdf_upload_{st.session_state.upload_key}",
        label_visibility="collapsed"
    )

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()

        with st.spinner(
            "Processing research paper..."
        ):

            try:

                pages, collection = process_pdf(
                    file_bytes
                )

                chat_id = create_chat(
                    uploaded_file.name,
                    collection,
                    pages
                )

                st.session_state.collection = (
                    collection
                )

                st.session_state.paper_name = (
                    uploaded_file.name
                )

                st.session_state.pages = pages

                st.success(
                    "Research paper processed successfully."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to process PDF: {e}"
                )


# ============================================================
# PAPER CHAT
# ============================================================

else:

    paper_name = current_chat[
        "paper_name"
    ]

    pages = current_chat[
        "pages"
    ]

    collection = current_chat[
        "collection"
    ]

    history = current_chat[
        "history"
    ]

    # --------------------------------------------------------
    # TOP BAR
    # --------------------------------------------------------

    title_col, action_col = st.columns(
        [5, 1]
    )

    with title_col:

        st.markdown(
            f"### {paper_name}"
        )

        total_words = sum(
            len(
                page["text"].split()
            )
            for page in pages
        )

        st.caption(
            f"{len(pages)} pages · {total_words:,} words"
        )

    with action_col:

        if st.button(
            "New paper",
            use_container_width=True
        ):

            new_chat()

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_col1, summary_col2 = st.columns(
        [1, 5]
    )

    with summary_col1:

        generate_summary_button = st.button(
            "Summary",
            use_container_width=True
        )

    with summary_col2:

        if current_chat["summary"]:

            with st.expander(
                "View paper summary"
            ):

                st.write(
                    current_chat["summary"]
                )

    if generate_summary_button:

        with st.spinner(
            "Generating paper summary..."
        ):

            try:

                summary = generate_summary(
                    pages
                )

                current_chat["summary"] = (
                    summary
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to generate summary: {e}"
                )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for item in history:

        question = item[
            "question"
        ]

        answer = item[
            "answer"
        ]

        chunks = item.get(
            "chunks",
            []
        )

        metadata = item.get(
            "metadata",
            []
        )

        distances = item.get(
            "distances",
            []
        )

        # Question LEFT
        question_col, answer_col = st.columns(
            [1, 2],
            gap="large"
        )

        with question_col:

            st.markdown(
                "**You**"
            )

            st.write(
                question
            )

        # Answer RIGHT
        with answer_col:

            st.markdown(
                "**PaperLens**"
            )

            st.write(
                answer
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            if chunks:

                with st.expander(
                    f"Sources · {len(chunks)} retrieved passages"
                ):

                    for i in range(
                        len(chunks)
                    ):

                        page = metadata[i].get(
                            "page",
                            "Unknown"
                        )

                        distance = distances[i]

                        st.markdown(
                            f"**Page {page} · Source {i + 1}**"
                        )

                        st.write(
                            chunks[i]
                        )

                        st.caption(
                            f"Relevance distance: {distance:.4f}"
                        )

                        if i < len(chunks) - 1:

                            st.divider()

        st.divider()

    # --------------------------------------------------------
    # EMPTY CHAT STATE
    # --------------------------------------------------------

    if not history:

        st.write("")

        st.markdown(
            "Start asking questions about the paper."
        )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask about this research paper..."
    )

    if question:

        question = question.strip()

        if question:

            with st.spinner(
                "Thinking..."
            ):

                try:

                    (
                        answer,
                        retrieved_chunks,
                        retrieved_metadata,
                        retrieved_ids,
                        distances
                    ) = ask_question(
                        question,
                        collection,
                        st.session_state.explanation_mode
                    )

                    current_chat[
                        "history"
                    ].append(
                        {
                            "question": question,

                            "answer": answer,

                            "chunks": retrieved_chunks,

                            "metadata": retrieved_metadata,

                            "ids": retrieved_ids,

                            "distances": distances
                        }
                    )

                    # Move this chat to top of Recents
                    chat_id = current_chat[
                        "id"
                    ]

                    if chat_id in st.session_state.recents:

                        st.session_state.recents.remove(
                            chat_id
                        )

                    st.session_state.recents.insert(
                        0,
                        chat_id
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {e}"
                    )
