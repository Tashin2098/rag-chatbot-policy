"""
RAG Policy Chatbot — ChatGPT-style multi-chat UI with per-chat memory and per-chat documents
Author: MD Tashin Rahman

Features (UI-only):
- Each chat keeps its own history + its own FAISS index file (isolation)
- Upload widget is inline in the chat area (not in sidebar)
- Sidebar shows "Chat 1, Chat 2, ..." for clear history navigation
- Main area shows "New Conversation" for empty chats (welcoming), then updates to first query text
- Welcome screen for brand-new chats; greetings work even before docs are uploaded
- No changes to ingestion/retrieval/generation logic
"""

import os
import uuid
import streamlit as st
from src.ingestion import DocumentIngestor
from src.vectorstore import FAISSVectorStore
from src.retrieval import Retriever
from src.generation import AnswerGenerator

st.set_page_config(page_title="Company Policy RAG Chatbot", page_icon="📚", layout="wide")


# ======================== HELPER FUNCTIONS ========================
def create_new_chat():
    """Create a new chat with its own FAISS index and empty history."""
    chat_id = str(uuid.uuid4())[:8]
    idx_path = f"faiss_{chat_id}.index"
    vs = FAISSVectorStore(index_path=idx_path)
    chat = {
        "id": chat_id,
        "sidebar_title": f"Chat {len(st.session_state.chats)+1}",  # Sidebar label
        "display_title": "New Conversation",  # Main area title
        "messages": [],
        "vectorstore_path": idx_path,
        "sources": []
    }
    st.session_state.chats.append(chat)
    st.session_state.active_chat_idx = len(st.session_state.chats) - 1
    st.session_state.vectorstore = vs
    st.session_state.retriever = Retriever(vs)
    st.session_state.chat_history = chat["messages"]


def switch_to_chat(idx: int):
    """Swap active UI state to a different chat (and load its FAISS index)."""
    st.session_state.active_chat_idx = idx
    active = st.session_state.chats[idx]
    st.session_state.vectorstore = FAISSVectorStore(index_path=active["vectorstore_path"])
    st.session_state.retriever = Retriever(st.session_state.vectorstore)
    st.session_state.chat_history = active["messages"]


# ======================== INITIALIZATION ========================
if "chats" not in st.session_state:
    st.session_state.chats = []
    st.session_state.active_chat_idx = 0

if "vectorstore" not in st.session_state:
    create_new_chat()
    try:
        st.session_state.generator = AnswerGenerator()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.session_state.ingestor = DocumentIngestor()
    st.session_state.show_context = False


# ======================== SIDEBAR ========================
with st.sidebar:
    st.header("💬 Chats")

    # List/select chats
    if st.session_state.chats:
        labels = [f"{c['sidebar_title']} ({c['id']})" for c in st.session_state.chats]
        sel = st.selectbox("Select chat", labels, index=st.session_state.active_chat_idx, label_visibility="collapsed")
        idx = labels.index(sel)
        if idx != st.session_state.active_chat_idx:
            switch_to_chat(idx)

    if st.button("＋ New chat", type="primary", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()

    # Chat settings
    with st.expander("⚙️ Settings"):
        st.checkbox("Show retrieved context", key="show_context")

    # Maintenance
    with st.expander("🛠️ Maintenance"):
        if st.button("Reset this chat index", use_container_width=True):
            active = st.session_state.chats[st.session_state.active_chat_idx]
            try:
                if os.path.exists(active["vectorstore_path"]):
                    os.remove(active["vectorstore_path"])
                if os.path.exists(active["vectorstore_path"] + ".meta"):
                    os.remove(active["vectorstore_path"] + ".meta")
                st.session_state.vectorstore = FAISSVectorStore(index_path=active["vectorstore_path"])
                st.session_state.retriever = Retriever(st.session_state.vectorstore)
                active["sources"] = []
                st.success("✅ Index cleared")
            except Exception as e:
                st.error(f"❌ {e}")

    # Show indexed files for this chat
    active = st.session_state.chats[st.session_state.active_chat_idx]
    if active["sources"]:
        with st.expander(f"📂 Files ({len(active['sources'])})"):
            for s in active["sources"]:
                st.write(f"• {s}")


# ======================== MAIN CHAT UI ========================
active = st.session_state.chats[st.session_state.active_chat_idx]
st.title(active["display_title"])
st.caption("💡 Your intelligent policy assistant. Upload, search, discover—answers powered by AI you can trust.")

is_empty_chat = len(active["messages"]) == 0

if is_empty_chat:
    # ======== WELCOME VIEW ========
    st.markdown("---")
    st.markdown("### Welcome to the Company Policy RAG Chatbot")
    st.markdown("""
    **Get started in 2 simple steps:**
    1. Upload your policy documents (PDF or TXT)
    2. Ask any question in plain language
    
    """)

    with st.expander("📤 Upload files to this chat", expanded=True):
        up_files = st.file_uploader(
            "Choose PDF or TXT files", 
            type=["pdf", "txt"], 
            accept_multiple_files=True, 
            key=f"u_{active['id']}"
        )
        if up_files and st.button("🚀 Process files", key=f"proc_{active['id']}", type="primary"):
            with st.spinner("Indexing files..."):
                os.makedirs("data", exist_ok=True)
                added = 0
                for f in up_files:
                    fp = os.path.join("data", f"{active['id']}_{f.name}")
                    with open(fp, "wb") as out:
                        out.write(f.getbuffer())
                    chunks, fname = st.session_state.ingestor.process_document(fp)
                    if chunks:
                        st.session_state.vectorstore.add_documents(chunks, fname)
                        if fname not in active["sources"]:
                            active["sources"].append(fname)
                        added += 1
                st.success(f"✅ Successfully added {added} file(s)!")
                st.rerun()

    

else:
    # ======== CONVERSATION VIEW ========
    # Render existing messages
    for message in active["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("📎 Sources"):
                    for src in message["sources"]:
                        st.write(f"• {src}")
            if message.get("context") and st.session_state.show_context:
                with st.expander("🔍 Retrieved context"):
                    st.write(message["context"])

    # Inline per-chat upload (collapsed)
    with st.expander("Add more files to this chat"):
        up_files = st.file_uploader(
            "Choose PDF or TXT files", 
            type=["pdf", "txt"], 
            accept_multiple_files=True, 
            key=f"u_{active['id']}"
        )
        if up_files and st.button("Process files", key=f"proc_{active['id']}"):
            with st.spinner("Indexing files..."):
                os.makedirs("data", exist_ok=True)
                added = 0
                for f in up_files:
                    fp = os.path.join("data", f"{active['id']}_{f.name}")
                    with open(fp, "wb") as out:
                        out.write(f.getbuffer())
                    chunks, fname = st.session_state.ingestor.process_document(fp)
                    if chunks:
                        st.session_state.vectorstore.add_documents(chunks, fname)
                        if fname not in active["sources"]:
                            active["sources"].append(fname)
                        added += 1
                st.success(f"✅ Successfully added {added} file(s)!")
                st.rerun()


# ======================== CHAT INPUT ========================
if query := st.chat_input("Ask about company policies..."):
    # ======== GREETING MODE (no docs required) ========
    if st.session_state.vectorstore.get_stats()["total_chunks"] == 0:
        qlow = query.strip().lower()
        if qlow in ("hi", "hello", "hey", "assalamualaikum", "salam", "good morning", "good afternoon"):
            active["messages"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                greeting = (
                    "Hello! Welcome to the Policy Chatbot.\n\n"
                    "To get started, upload a policy PDF or TXT file using the panel above. "
                    "Once uploaded, I'll answer any questions based on your documents with full citations."
                )
                st.write(greeting)
                active["messages"].append({"role": "assistant", "content": greeting})
            st.rerun()
        else:
            st.warning("⚠️ Please upload policy documents first to ask questions.")
            st.stop()

    # ======== NORMAL GROUNDED QA FLOW ========
    active["messages"].append({"role": "user", "content": query})

    # Update main title after first message (like ChatGPT)
    if len(active["messages"]) == 1:
        active["display_title"] = query[:40] + "..." if len(query) > 40 else query

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            context, citations = st.session_state.retriever.retrieve(query, top_k=3)
            answer = st.session_state.generator.generate_answer(query, context)

            st.write(answer)
            with st.expander("📎 Sources"):
                for i, c in enumerate(citations, 1):
                    st.write(f"{i}. {c}")
            if st.session_state.show_context:
                with st.expander("🔍 Retrieved context"):
                    st.write(context)

            active["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": citations,
                "context": context if st.session_state.show_context else None
            })
    st.rerun()


# ======================== FOOTER ========================
st.divider()

