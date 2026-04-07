import streamlit as st
import json
import os
from pathlib import Path
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Configure page
st.set_page_config(
    page_title="Nftify - NFT Knowledge Base",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling with premium dark theme
st.markdown(
    """
    <style>
    * {
        margin: 0;
        padding: 0;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    html, body {
        background: linear-gradient(135deg, #0a0e27 0%, #0d1120 50%, #0a0e27 100%);
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .stApp {
        background: transparent;
    }

    .app-shell {
        background: linear-gradient(180deg, rgba(10,14,39,0.4) 0%, rgba(15,18,45,0.5) 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 28px;
        padding: 32px;
        box-shadow: 0 25px 80px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 12px;
    }

    .brand-badge {
        width: 56px;
        height: 56px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        box-shadow: 0 15px 40px rgba(139, 92, 246, 0.4), inset 0 1px 1px rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .brand-text h1 {
        margin: 0;
        font-size: 2.2rem;
        line-height: 1.1;
        color: #f3f4f6;
        letter-spacing: -0.03em;
        font-weight: 700;
    }

    .brand-text p {
        margin: 6px 0 0 0;
        color: #c5cae9;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    .panel {
        background: linear-gradient(180deg, rgba(30, 30, 60, 0.3) 0%, rgba(25, 25, 50, 0.2) 100%);
        border: 1px solid rgba(139, 92, 246, 0.12);
        border-radius: 22px;
        padding: 24px;
        height: 100%;
        backdrop-filter: blur(8px);
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.05);
    }

    .section-label {
        color: #a78bfa;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 14px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }

    .answer-card {
        background: linear-gradient(180deg, rgba(139, 92, 246, 0.08) 0%, rgba(99, 102, 241, 0.04) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 22px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.15), inset 0 1px 1px rgba(255,255,255,0.08);
        line-height: 1.7;
        color: #e5e7eb;
    }

    .answer-card p {
        margin: 10px 0;
    }

    .answer-card strong {
        color: #f3f4f6;
    }

    .answer-card ul, .answer-card ol {
        margin: 12px 0 12px 20px;
    }

    .answer-card li {
        margin: 6px 0;
    }

    .source-card {
        background: linear-gradient(180deg, rgba(99, 102, 241, 0.05) 0%, rgba(55, 65, 81, 0.04) 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }

    .source-card:hover {
        background: linear-gradient(180deg, rgba(139, 92, 246, 0.1) 0%, rgba(99, 102, 241, 0.08) 100%);
        border-color: rgba(139, 92, 246, 0.25);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.15), inset 0 1px 1px rgba(255,255,255,0.08);
    }

    .source-title {
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }

    .source-meta {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-bottom: 10px;
        word-break: break-all;
    }

    .source-snippet {
        color: #d1d5db;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .empty-state {
        text-align: center;
        padding: 60px 30px;
        color: #9ca3af;
        background: linear-gradient(180deg, rgba(30, 30, 60, 0.2) 0%, rgba(25, 25, 50, 0.1) 100%);
        border-radius: 20px;
        border: 1px dashed rgba(139, 92, 246, 0.2);
    }

    .empty-state-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #d1d5db;
        margin-bottom: 8px;
    }

    .empty-state-desc {
        font-size: 0.95rem;
        color: #9ca3af;
    }

    .footer-note {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 14px;
        font-style: italic;
    }

    .stTextArea textarea {
        border-radius: 16px !important;
        min-height: 140px !important;
        background: linear-gradient(180deg, rgba(30, 30, 60, 0.4) 0%, rgba(25, 25, 50, 0.3) 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        color: #e5e7eb !important;
        font-size: 0.95rem !important;
        font-family: inherit !important;
        padding: 14px !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(139, 92, 246, 0.4) !important;
        background: linear-gradient(180deg, rgba(30, 30, 60, 0.5) 0%, rgba(25, 25, 50, 0.4) 100%) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
    }

    .stTextArea label {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stButton"] > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1.5rem !important;
        margin-top: 1.8rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 10px 25px rgba(139, 92, 246, 0.3) !important;
        color: white !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #9d67ff 0%, #8d4aff 100%) !important;
        box-shadow: 0 15px 35px rgba(139, 92, 246, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    .pill {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(99, 102, 241, 0.1));
        border: 1px solid rgba(139, 92, 246, 0.25);
        color: #c7d2fe;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.12);
    }

    a {
        color: #a78bfa !important;
        text-decoration: none;
        transition: color 0.3s ease !important;
        font-weight: 500;
    }

    a:hover {
        color: #c4b5fd !important;
    }

    .stSlider {
        margin-bottom: 1rem;
    }

    .stSlider label {
        color: #d1d5db !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    div[data-testid="stExpander"] {
        border-radius: 16px !important;
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
        background: linear-gradient(180deg, rgba(30, 30, 60, 0.2) 0%, rgba(25, 25, 50, 0.1) 100%) !important;
    }

    div[data-testid="stExpander"] > div {
        background: transparent !important;
    }

    .info-box {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 14px !important;
        color: #e0e7ff !important;
    }

    .warning-box {
        background: rgba(249, 115, 22, 0.1) !important;
        border: 1px solid rgba(249, 115, 22, 0.2) !important;
        border-radius: 14px !important;
        color: #fed7aa !important;
    }

    .error-box {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
        border-radius: 14px !important;
        color: #fca5a5 !important;
    }

    div[data-testid="stForm"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state
if "pages_data" not in st.session_state:
    st.session_state.pages_data = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []


def load_jsonl_file(file):
    """Load and parse JSONL file"""
    try:
        pages = []
        content = file.read().decode("utf-8")
        for line in content.strip().split("\n"):
            if line.strip():
                pages.append(json.loads(line))
        return pages
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []


def search_relevant_pages(query: str, pages: list, top_k: int = 3) -> list:
    """Simple keyword-based search for relevant pages"""
    query_lower = query.lower()
    scored_pages = []

    for page in pages:
        content = page.get("content", "").lower()
        title = page.get("title", "").lower()

        title_score = title.count(query_lower)
        content_score = content.count(query_lower) * 0.5

        total_score = title_score + content_score

        if total_score > 0:
            scored_pages.append((page, total_score))

    scored_pages.sort(key=lambda x: x[1], reverse=True)
    return [page for page, _ in scored_pages[:top_k]]


def generate_answer(query: str, sources: list) -> str:
    """Generate answer using Gemini API"""
    if not GOOGLE_API_KEY:
        return "Please set the GOOGLE_API_KEY environment variable to use the AI features."

    try:
        model = GenerativeModel("gemini-2.5-flash")

        sources_text = "\n\n".join(
            [f"Source: {s.get('title', 'Unknown')}\n{s.get('content', '')}" for s in sources]
        )

        prompt = f"""Based on the following sources about NFTs, answer the user's question:

Sources:
{sources_text}

User Question: {query}

Please provide a detailed and helpful answer based on the sources provided."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"


def get_suggested_questions(pages: list) -> list:
    """Generate suggested questions from the loaded data"""
    suggestions = [
        "What is an NFT?",
        "How do NFT smart contracts work?",
        "What are the main use cases for NFTs?",
        "How do you mint an NFT?",
        "What is gas and how does it relate to NFTs?",
    ]
    return suggestions


# Main UI
with st.container():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown(
            '<div class="brand"><div class="brand-badge">💎</div><div class="brand-text"><h1>Nftify</h1><p>Your NFT Knowledge Base Assistant</p></div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# Left Panel - Query Input
with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📤 Upload Knowledge Base</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload JSONL file", type="jsonl", key="file_uploader")
    if uploaded_file is not None:
        st.session_state.pages_data = load_jsonl_file(uploaded_file)
        st.success(f"✓ Loaded {len(st.session_state.pages_data)} pages")

    st.markdown('<p class="section-label" style="margin-top: 20px;">❓ Ask a Question</p>', unsafe_allow_html=True)

    query = st.text_area(
        "Enter your NFT question",
        height=140,
        placeholder="e.g., How do NFT smart contracts work?",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        search_button = st.button("🔍 Search & Answer", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🔄 Clear", use_container_width=True)

    if clear_button:
        st.session_state.current_answer = None
        st.session_state.current_sources = []
        st.rerun()

    if search_button and query and st.session_state.pages_data:
        with st.spinner("Searching knowledge base and generating answer..."):
            sources = search_relevant_pages(query, st.session_state.pages_data, top_k=3)
            if sources:
                st.session_state.current_sources = sources
                answer = generate_answer(query, sources)
                st.session_state.current_answer = answer
            else:
                st.warning("No relevant sources found for your query.")

    st.markdown("</div>", unsafe_allow_html=True)

# Right Panel - Results
with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if st.session_state.pages_data:
        st.markdown('<p class="section-label">💡 Suggested Questions</p>', unsafe_allow_html=True)
        suggestions = get_suggested_questions(st.session_state.pages_data)
        for suggestion in suggestions:
            st.markdown(f'<span class="pill">{suggestion}</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-title">No Data Loaded</div><div class="empty-state-desc">Upload a JSONL file to get started</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.current_answer:
        st.markdown('<p class="section-label" style="margin-top: 20px;">🤖 Answer</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{st.session_state.current_answer}</div>', unsafe_allow_html=True)

    if st.session_state.current_sources:
        st.markdown('<p class="section-label" style="margin-top: 20px;">📚 Sources</p>', unsafe_allow_html=True)
        for source in st.session_state.current_sources:
            title = source.get("title", "Unknown Source")
            content = source.get("content", "")
            snippet = content[:200] + "..." if len(content) > 200 else content

            st.markdown(
                f'''
                <div class="source-card">
                    <div class="source-title">{title}</div>
                    <div class="source-snippet">{snippet}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    '<p class="footer-note">💡 Tip: Upload a JSONL file with NFT knowledge, ask questions, and get AI-powered answers with sources</p>',
    unsafe_allow_html=True,
)
