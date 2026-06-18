"""
frontend/app.py
================
NexusIQ — Streamlit Demo Interface (Phase 1)

This is the visual face of NexusIQ for demos and development.
Streamlit is perfect for AI demos because:
1. Pure Python — no JavaScript/HTML needed
2. Hot-reload on save — see changes instantly
3. Built-in widgets: file uploaders, chat interfaces, charts
4. Deployable to Streamlit Cloud in minutes (free)

HOW TO RUN:
  streamlit run frontend/app.py

ARCHITECTURE NOTE:
The Streamlit frontend talks to the FastAPI backend via HTTP.
This separation means:
- The UI is replaceable (swap Streamlit for Next.js in Phase 3)
- The API can be called by other clients (mobile apps, integrations)
- Authentication happens at the API layer, not the UI layer

In Phase 3, we'll replace Streamlit with a Next.js frontend
for a production-quality user experience.
"""

import streamlit as st
import httpx
import time
import json
from pathlib import Path

# ─── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="NexusIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Configuration ───────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"


# ─── Session State ──────────────────────────────────────────
# st.session_state persists across reruns (Streamlit reruns on every interaction).
# We use it to store:
# - Authentication token (so user doesn't re-login on each click)
# - Chat history (to display the conversation)
# - Current user info

def init_session_state():
    defaults = {
        "token": None,
        "user": None,
        "chat_history": [],
        "show_sources": True,
        "page": "chat",  # "chat" | "documents" | "analytics"
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# ─── API Client ─────────────────────────────────────────────
def api_headers() -> dict:
    """Returns auth headers for API requests."""
    return {
        "Authorization": f"Bearer {st.session_state.token}",
        "Content-Type": "application/json",
    }


def api_call(method: str, endpoint: str, **kwargs) -> dict | None:
    """
    Makes an authenticated API call with error handling.

    Centralized API calls mean:
    - One place to handle auth errors (redirect to login)
    - One place to add request logging
    - Consistent error messaging
    """
    try:
        url = f"{API_BASE}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers.update(api_headers())

        response = httpx.request(method, url, headers=headers, timeout=60.0, **kwargs)

        if response.status_code == 401:
            st.session_state.token = None
            st.session_state.user = None
            st.error("Session expired. Please log in again.")
            st.rerun()

        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.text}")
        return None
    except httpx.ConnectError:
        st.error("Cannot connect to NexusIQ API. Is the server running? (`uvicorn backend.app.main:app --reload`)")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ─── Authentication Pages ────────────────────────────────────
def render_login_page():
    """Login / Register page shown when user is not authenticated."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("# 🧠 NexusIQ")
        st.markdown("*Enterprise Intelligence Platform*")
        st.divider()

        tab_login, tab_register = st.tabs(["Sign In", "Register"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@company.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if submitted:
                    email = "jane@company.com"
                    password = "Abc@123123"
                    if not email or not password:
                        st.error("Please fill in all fields")
                    else:
                        with st.spinner("Signing in..."):
                            try:
                                response = httpx.post(
                                    f"{API_BASE}/auth/login",
                                    json={"email": email, "password": password},
                                    timeout=10.0,
                                )
                                if response.status_code == 200:
                                    data = response.json()
                                    st.session_state.token = data["access_token"]
                                    st.session_state.user = data["user"]
                                    st.success("Welcome back!")
                                    st.rerun()
                                else:
                                    st.error("Invalid email or password")
                            except httpx.ConnectError:
                                st.error("Cannot connect to server. Is NexusIQ running?")

        with tab_register:
            with st.form("register_form"):
                name = st.text_input("Full Name", placeholder="Jane Smith")
                email = st.text_input("Email", placeholder="jane@company.com", key="reg_email")
                password = st.text_input("Password", type="password", key="reg_password",
                                        help="Min 8 characters, 1 uppercase, 1 number")
                submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if submitted:
                    name = "Jane Smith"
                    email = "jane@company.com"
                    password = "Abc@123123"
                    try:
                        response = httpx.post(
                            f"{API_BASE}/auth/register",
                            json={"email": email, "full_name": name, "password": password},
                            timeout=10.0,
                        )
                        if response.status_code == 201:
                            st.success("Account created! Please sign in.")
                        else:
                            detail = response.json().get("detail", "Registration failed")
                            st.error(detail)
                    except Exception as e:
                        st.error(f"Error: {e}")


# ─── Sidebar ─────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 🧠 NexusIQ")

        if st.session_state.user:
            user = st.session_state.user
            st.markdown(f"**{user['full_name']}**")
            st.markdown(f"*{user['email']}*")
            st.markdown(f"`{user['role'].upper()}`")
            st.divider()

        # Navigation
        pages = {
            "💬 Chat": "chat",
            "📄 Documents": "documents",
            "📊 Analytics": "analytics",
        }
        for label, page_id in pages.items():
            if st.button(label, use_container_width=True,
                        type="primary" if st.session_state.page == page_id else "secondary"):
                st.session_state.page = page_id
                st.rerun()

        st.divider()

        # Settings
        st.session_state.show_sources = st.toggle(
            "Show source citations",
            value=st.session_state.show_sources
        )

        st.divider()
        if st.button("Sign Out", use_container_width=True):
            for key in ["token", "user", "chat_history"]:
                st.session_state[key] = None if key != "chat_history" else []
            st.rerun()


# ─── Chat Page ───────────────────────────────────────────────
def render_chat_page():
    st.markdown("## 💬 Ask NexusIQ")
    st.markdown("*Ask anything about your knowledge base. Every answer is cited.*")

    # ─── Display chat history ────────────────────────────────
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if enabled and present
            if message["role"] == "assistant" and st.session_state.show_sources:
                sources = message.get("sources", [])
                if sources:
                    with st.expander(f"📚 {len(sources)} source(s) used", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {source['document_name']}")
                            if source.get("page_number"):
                                st.markdown(f"*Page {source['page_number']}*")
                            if source.get("section_header"):
                                st.markdown(f"*Section: {source['section_header']}*")
                            st.markdown(f"Relevance: `{source['relevance_score']:.2f}`")
                            with st.expander("View chunk content"):
                                st.text(source["chunk_content"])
                            st.divider()

                # Show quality metrics
                meta = message.get("meta", {})
                if meta:
                    cols = st.columns(4)
                    cols[0].metric("Latency", f"{meta.get('latency_ms', 0)}ms")
                    cols[1].metric("Tokens", f"{meta.get('tokens_used', 0):,}")
                    cols[2].metric("Sources", len(sources))
                    if meta.get("confidence"):
                        cols[3].metric("Confidence", f"{meta.get('confidence', 0):.0%}")

    # ─── Chat input ─────────────────────────────────────────
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        # Display it immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating answer..."):
                start = time.time()
                result = api_call("POST", "/queries/ask", json={"question": question})

            if result:
                st.markdown(result["answer"])

                # Add to history with sources
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "meta": {
                        "latency_ms": result.get("latency_ms"),
                        "tokens_used": result.get("tokens_used"),
                        "confidence": result.get("confidence_score"),
                    },
                })

                # Show sources inline
                if st.session_state.show_sources and result.get("sources"):
                    with st.expander(f"📚 {len(result['sources'])} source(s)", expanded=True):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**[{i}] {source['document_name']}**" +
                                       (f" — Page {source['page_number']}" if source.get('page_number') else ""))
                            st.caption(source.get("chunk_content", "")[:200] + "...")
                            st.divider()
            else:
                error_msg = "Sorry, I couldn't generate an answer. Please try again."
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                })

        st.rerun()


# ─── Documents Page ──────────────────────────────────────────
def render_documents_page():
    st.markdown("## 📄 Knowledge Base")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md", "docx"],
            help="PDF, Markdown, TXT, or Word documents",
        )
        description = st.text_area("Description (optional)", height=80)

        if uploaded_file and st.button("Upload & Index", type="primary"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {}
                    if description:
                        data["description"] = description

                    response = httpx.post(
                        f"{API_BASE}/documents/upload",
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        files=files,
                        data=data,
                        timeout=120.0,
                    )

                    if response.status_code == 202:
                        result = response.json()
                        st.success(f"✅ '{uploaded_file.name}' ingested!")
                        st.info(f"Created {result.get('chunks_created', 0)} searchable chunks")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.text}")

                except httpx.ConnectError:
                    st.error("Cannot connect to server")

    with col1:
        st.markdown("### Indexed Documents")

        result = api_call("GET", "/documents/")
        if result:
            docs = result.get("documents", [])

            if not docs:
                st.info("No documents yet. Upload some documents to get started!")
            else:
                st.caption(f"Total: {result.get('total', 0)} documents")

                for doc in docs:
                    status_emoji = {
                        "indexed": "✅",
                        "processing": "⏳",
                        "pending": "🔄",
                        "failed": "❌",
                    }.get(doc["status"], "❓")

                    with st.expander(f"{status_emoji} {doc['filename']} — {doc['chunk_count']} chunks"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Status", doc["status"].title())
                        col_b.metric("Chunks", doc["chunk_count"])
                        col_c.metric("Tokens", f"{doc.get('total_tokens', 0):,}")

                        if doc.get("description"):
                            st.caption(doc["description"])

                        if doc.get("error_message"):
                            st.error(f"Error: {doc['error_message']}")


# ─── Analytics Page ──────────────────────────────────────────
def render_analytics_page():
    st.markdown("## 📊 Analytics")

    # Query history
    result = api_call("GET", "/queries/history")
    if result:
        if not result:
            st.info("No queries yet. Ask something in the Chat tab!")
            return

        st.markdown(f"**{len(result)} recent queries**")

        for query in result:
            with st.expander(f"Q: {query['question'][:80]}..."):
                st.markdown(f"**Answer preview:** {(query.get('answer') or '')[:200]}...")
                cols = st.columns(4)
                cols[0].metric("Status", query["status"])
                cols[1].metric("Latency", f"{query.get('latency_ms', 0)}ms")
                if query.get("faithfulness_score"):
                    cols[2].metric("Faithfulness", f"{query['faithfulness_score']:.2f}")
                st.caption(f"Asked: {query['created_at']}")


# ─── Main App ────────────────────────────────────────────────
def main():
    # Not authenticated → show login
    if not st.session_state.token:
        render_login_page()
        return

    # Authenticated → show app
    render_sidebar()

    page = st.session_state.page
    if page == "chat":
        render_chat_page()
    elif page == "documents":
        render_documents_page()
    elif page == "analytics":
        render_analytics_page()


if __name__ == "__main__":
    main()
