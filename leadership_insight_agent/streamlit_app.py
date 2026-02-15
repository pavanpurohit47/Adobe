import sys
import subprocess
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import typer


# ---------- ENV (Python 3.13 safe) ----------
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", override=True)

st.set_page_config(page_title="Leadership Insight Agent", layout="wide")
st.title("📊 Leadership Insight Agent")
st.caption("RAG Chatbot over your documents (Index → Ask → Evidence-grounded response)")

# ---------- SIDEBAR SETTINGS ----------
with st.sidebar:
    st.header("⚙️ Settings")

    docs_dir = st.text_input("Docs folder", value=str(ROOT / "docs"))
    index_dir = st.text_input("Index folder", value=str(ROOT / "indices" / "company"))

    st.markdown("---")
    st.subheader("Actions")

    if st.button("🧱 Build / Rebuild Index", use_container_width=True):
        try:
            with st.spinner("Indexing documents (chunking + embeddings + FAISS)..."):
                out = subprocess.run(
                    [sys.executable, "-m", "leadership_agent.cli", "index", "--docs", docs_dir, "--out", index_dir],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True
                )
            if out.returncode != 0:
                st.error(out.stderr or out.stdout)
            else:
                st.success("Index built ✅")
                st.code(out.stdout.strip())
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_error = None
        st.rerun()

    st.markdown("---")
    st.subheader("LLM (from .env)")
    st.write("BASE_URL:", st.session_state.get("base_url", ""))
    st.write("MODEL:", st.session_state.get("model", ""))
    st.write("API KEY set:", bool(st.session_state.get("api_key_set", False)))

# Store env display once
if "env_loaded" not in st.session_state:
    st.session_state.env_loaded = True
    import os
    st.session_state.base_url = os.getenv("OPENAI_BASE_URL", "")
    st.session_state.model = os.getenv("OPENAI_MODEL", "")
    st.session_state.api_key_set = bool(os.getenv("OPENAI_API_KEY"))

# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role":"user/assistant", "content": "..."}
if "last_error" not in st.session_state:
    st.session_state.last_error = None

def ask_cli(question: str) -> str:
    """Call your existing CLI and return plain text answer."""
    cmd = [sys.executable, "-m", "leadership_agent.cli", "ask", "--index", index_dir, "--question", question]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return r.stdout.strip()

# ---------- RENDER CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- INPUT (ChatGPT-style) ----------
user_q = st.chat_input("Ask a question… (e.g., What are key risks highlighted?)")

if user_q:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(user_q)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence + generating answer..."):
                answer = ask_cli(user_q)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_error = None

        except Exception as e:
            err = str(e)
            st.session_state.last_error = err
            st.error(err)

# Optional: show last error below (if you want)
if st.session_state.last_error:
    st.warning("Last error occurred. Check your index path/docs and LLM configuration.")
