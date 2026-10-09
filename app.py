import html
import re
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from io import BytesIO

import streamlit as st
import chromadb
from groq import Groq
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#2F5D8A"
LIGHT_BLUE = "#E1ECF7"

PRIMARY_MODEL = "gemini-3-flash-preview"

GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]

DISPLAY_TZ = timezone(timedelta(hours=5, minutes=30), "IST")

MODES = ["Simple", "Technical"]

MODE_HINTS = {
    "Simple": "Plain, beginner-friendly answers.",
    "Technical": "Detailed answers with research terminology.",
}

STYLE_INSTRUCTIONS = {
    "Simple": (
        "Explain the answer in simple English. Use beginner-friendly "
        "language and avoid unnecessary technical terminology."
    ),
    "Technical": (
        "Give a detailed technical explanation using appropriate "
        "research and computer science terminology."
    ),
}


# ============================================================
# CSS — PAPERLENS DESIGN
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --pl-blue: #2F5D8A;
    --pl-light-blue: #E1ECF7;
}

.stApp {
    font-family: 'Public Sans', system-ui, sans-serif;
}

footer, #MainMenu {
    display: none;
}

/* Hide toolbar actions but preserve the header and sidebar toggle */
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"] {
    display: none !important;
}

/* Preserve Streamlit header */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Keep sidebar reopening controls visible */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

.block-container {
    max-width: 880px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* Sidebar */
.pl-logo-name {
    font: 600 1.45rem 'Newsreader', Georgia, serif;
    line-height: 1.1;
    margin: 0 0 .3rem 0;
}

[data-testid="stSidebar"] [class*="st-key-recent_"] button,
[data-testid="stSidebar"] [class*="st-key-pinned_"] button {
    background: transparent;
    border: none;
    border-radius: 8px;
    justify-content: flex-start;
    text-align: left;
    font-weight: 500;
    padding: .5rem .75rem;
    min-height: 0;
    box-shadow: none;
}

[data-testid="stSidebar"] [class*="st-key-recent_"] button:hover,
[data-testid="stSidebar"] [class*="st-key-pinned_"] button:hover {
    background: rgba(47,93,138,.08);
}

[data-testid="stSidebar"] [class*="st-key-recent_"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] [class*="st-key-pinned_"] [data-testid="stBaseButton-primary"] {
    background: #E6EEF7;
    color: #1B2430;
    box-shadow: inset 3px 0 0 #2F5D8A;
}

.st-key-new_chat_btn button {
    background: #2F5D8A !important;
    color: white !important;
    border: none !important;
    justify-content: center !important;
    font-weight: 600;
    border-radius: 9px;
}

.st-key-new_chat_btn button:hover {
    background: #254B72 !important;
}

[data-testid="stSidebarHeader"] {
    height: auto;
    padding: .6rem 1rem 0 1rem;
}

[data-testid="stSidebarUserContent"] {
    padding-top: .3rem;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: .6rem;
}

[data-testid="stSidebar"] .stButton button > div {
    justify-content: flex-start;
    width: 100%;
}

[data-testid="stSidebar"] .stButton button p {
    text-align: left;
}

.st-key-new_chat_btn button > div {
    justify-content: center !important;
}

/* Chat actions */
[class*="st-key-act_"] button {
    background: transparent;
    border: none;
    box-shadow: none;
    border-radius: 8px;
    padding: .4rem .75rem;
    min-height: 0;
    font-weight: 500;
    justify-content: flex-start;
}

[class*="st-key-act_"] button:hover {
    background: rgba(47,93,138,.08);
}

.st-key-act_delete button,
.st-key-act_delete_yes button {
    color: #B3382C;
}

.st-key-act_delete button:hover {
    background: rgba(179,56,44,.08);
}

.st-key-act_delete_yes button,
.st-key-act_delete_no button {
    justify-content: center !important;
    border: 1px solid #D5DDE6 !important;
}

/* Mode buttons */
[class*="st-key-mode_"] [data-testid="stBaseButton-primary"] {
    background: #2F5D8A !important;
    border: 1px solid #2F5D8A !important;
    color: #fff !important;
}

[class*="st-key-mode_"] [data-testid="stBaseButton-primary"] p {
    color: #fff !important;
}

[class*="st-key-mode_"] [data-testid="stBaseButton-secondary"] {
    background: #fff !important;
    border: 1px solid #D5DDE6 !important;
    color: #1B2430 !important;
}

[class*="st-key-mode_"] [data-testid="stBaseButton-secondary"]:hover {
    border-color: #2F5D8A !important;
    color: #2F5D8A !important;
}

[class*="st-key-mode_"] button > div {
    justify-content: center !important;
}

[class*="st-key-mode_"] button p {
    text-align: center !important;
}

/* General text fields */
[data-testid="stTextInput"] input:focus {
    border-color: #2F5D8A !important;
    box-shadow: 0 0 0 1px #2F5D8A !important;
}

/* ========================================================
   CHAT TYPING BAR — NO RED FOCUS OUTLINE
   ======================================================== */

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: #FFFFFF !important;
    border: 1px solid #D5DDE6 !important;
    border-radius: 18px !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] {
    box-shadow: 0 2px 10px rgba(47, 93, 138, 0.08) !important;
    padding-right: 6px !important;
}

[data-testid="stChatInput"]:hover,
[data-testid="stChatInput"]:focus,
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    background: #FFFFFF !important;
    border-color: #9DB6CF !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    caret-color: #2F5D8A !important;
}

[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}


/* ========================================================
   SEND BUTTON — CLEAN ROUNDED BUTTON WITH REAL ICON
   ======================================================== */

[data-testid="stChatInputSubmitButton"] {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    padding: 0 !important;
    margin: 0 2px !important;
    border: none !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #3A6FA3 0%, #2F5D8A 55%, #254B72 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 8px rgba(47, 93, 138, 0.35) !important;
    outline: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease !important;
}

/* Show the real icon, in white */
[data-testid="stChatInputSubmitButton"] svg {
    display: block !important;
    width: 20px !important;
    height: 20px !important;
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

[data-testid="stChatInputSubmitButton"] svg path {
    fill: #FFFFFF !important;
}

/* Remove any leftover pseudo-element arrow */
[data-testid="stChatInputSubmitButton"]::before,
[data-testid="stChatInputSubmitButton"]::after {
    content: none !important;
    display: none !important;
}

/* Hover: lift slightly */
[data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 14px rgba(47, 93, 138, 0.42) !important;
    filter: brightness(1.06) !important;
}

/* Click: press down */
[data-testid="stChatInputSubmitButton"]:active:not(:disabled) {
    transform: translateY(0) scale(0.95) !important;
    box-shadow: 0 2px 5px rgba(47, 93, 138, 0.3) !important;
}

/* Disabled (empty input): soft and faded */
[data-testid="stChatInputSubmitButton"]:disabled {
    background: #C9D6E4 !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    opacity: 1 !important;
}

[data-testid="stChatInputSubmitButton"]:disabled svg,
[data-testid="stChatInputSubmitButton"]:disabled svg path {
    fill: #FFFFFF !important;
    opacity: .85;
}

/* Remove send-button focus outline */
[data-testid="stChatInputSubmitButton"]:focus,
[data-testid="stChatInputSubmitButton"]:focus-visible {
    outline: none !important;
    border: none !important;
}


/* Paper title */
.pl-title {
    font: 600 1.55rem 'Newsreader', Georgia, serif;
    margin: 0;
    line-height: 1.25;
}

.pl-meta {
    color: #66758a;
    font-size: .85rem;
    margin: .2rem 0 .8rem 0;
}

/* Chat messages */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none;
}

[data-testid="stChatMessage"] {
    width: fit-content;
    max-width: 88%;
    border-radius: 16px;
    padding: .8rem 1.1rem;
    background: #F4F7FA;
    border: 1px solid #E3E9F0;
    margin-right: auto;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto;
    margin-right: 0;
    max-width: 72%;
    background: #E1ECF7;
    border-color: #C9DCEF;
    border-bottom-right-radius: 4px;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-bottom-left-radius: 4px;
    background: #FFFFFF;
    border-color: #DDE5EE;
}

/* Citations */
.pl-cites {
    color: #66758a;
    font-size: .82rem;
    margin-top: .4rem;
}

/* Empty chat */
.pl-empty {
    text-align: center;
    padding: 3rem 1rem 1rem 1rem;
}

.pl-empty b {
    font: 600 1.5rem 'Newsreader', Georgia, serif;
}

.pl-empty span {
    display: block;
    color: #66758a;
    margin-top: .3rem;
}

/* PDF upload screen */
.pl-hero-title {
    font: 600 2.8rem 'Newsreader', Georgia, serif;
    line-height: 1.12;
    margin: 2rem 0 .5rem 0;
}

.pl-hero-sub {
    font-size: 1.05rem;
    color:
