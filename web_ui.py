"""
Web GUI for the assistant using Streamlit.
Run with: streamlit run web_ui.py
"""

import streamlit as st
import json
import os
import time
from pathlib import Path
from assistant import respond, analyze_and_remember, create_file_from_prompt, show_memory, mem

# Page config
st.set_page_config(
    page_title="Spring-2026 Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for UI feedback
if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = None

st.title("🤖 Spring-2026 Assistant")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    mode = st.radio(
        "Select Mode:",
        ["Chat", "Analyze File", "Create File", "Memory", "Help"]
    )
    
    st.divider()
    st.subheader("⚙️ Settings")
    
    use_fast_mode = st.checkbox(
        "⚡ Fast Mode (instant responses)",
        value=False,
        help="Skip LLM and use instant fallback responses"
    )
    
    model_choice = st.selectbox(
        "LLM Model:",
        ["Smart (Mistral 7B - 20-45s)", "Fast (Phi 2.5 - 5-10s)", "Powerful (Custom)"],
        help="Smarter model takes longer but gives better responses",
        index=0  # Smart by default
    )
    
    # Map choice to env var
    model_map = {
        "Smart (Mistral 7B - 20-45s)": "Mistral 7B",
        "Fast (Phi 2.5 - 5-10s)": "Phi 2.5",
        "Powerful (Custom)": os.environ.get("GPT4ALL_MODEL", "Mistral 7B")
    }
    os.environ["GPT4ALL_MODEL"] = model_map[model_choice]
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Load memory info at top
current_memory = show_memory()
num_messages = len(current_memory.get("messages", []))
num_analyses = len(current_memory.get("analyses", []))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Messages", num_messages)
with col2:
    st.metric("Files Analyzed", num_analyses)
with col3:
    if st.button("Clear Memory"):
        if os.path.exists(".assistant_memory.json"):
            os.remove(".assistant_memory.json")
            st.success("Memory cleared!")
            st.rerun()

st.divider()

# MODE: Chat
if mode == "Chat":
    st.header("💬 Chat with Assistant")
    
    # Show recent messages
    with st.expander("📜 Conversation History", expanded=False):
        messages = current_memory.get("messages", [])
        if messages:
            for msg in messages[-10:]:  # Show last 10
                role = msg.get("role", "unknown").upper()
                text = msg.get("text", "")
                if role == "USER":
                    st.info(f"**You:** {text}")
                elif role == "ASSISTANT":
                    st.success(f"**Assistant:** {text}")
                else:
                    st.caption(f"{role}: {text}")
        else:
            st.caption("No messages yet")
    
    # Chat input
    prompt = st.text_area("Ask the assistant:", placeholder="What would you like to know?", height=100)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        send_btn = st.button("Send", use_container_width=True, type="primary")
    with col2:
        if st.session_state.last_response_time:
            st.caption(f"⏱️ {st.session_state.last_response_time:.1f}s")
    
    if send_btn:
        if prompt.strip():
            if use_fast_mode:
                with st.spinner("⚡ Generating response..."):
                    try:
                        start = time.time()
                        from assistant import _fallback_response, _build_context_summary
                        ctx = _build_context_summary()
                        response = _fallback_response(prompt, ctx)
                        elapsed = time.time() - start
                        st.session_state.last_response_time = elapsed
                        
                        st.success("✅ Response received!")
                        st.markdown(f"**Assistant:** {response}")
                        st.caption(f"⏱️ Generated in {elapsed:.1f}s")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                with st.spinner("🤔 Thinking deeply (this may take 20-45 seconds)..."):
                    try:
                        start = time.time()
                        response = respond(prompt)
                        elapsed = time.time() - start
                        st.session_state.last_response_time = elapsed
                        
                        st.success("✅ Response received!")
                        st.markdown(f"**Assistant:** {response}")
                        st.caption(f"⏱️ Generated in {elapsed:.1f}s")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a message")


# MODE: Analyze File
elif mode == "Analyze File":
    st.header("📄 Analyze File")
    
    uploaded_file = st.file_uploader("Choose a file to analyze", type=["pdf", "txt", "py", "md", "json"])
    
    if uploaded_file:
        # Save temporarily and analyze
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analyze", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    analysis = analyze_and_remember(temp_path)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Words", analysis.get("words", 0))
                    with col2:
                        st.metric("Lines", analysis.get("lines", 0))
                    with col3:
                        st.metric("Characters", analysis.get("chars", 0))
                    
                    st.subheader("Top Keywords")
                    top_words = analysis.get("top_words", [])
                    if top_words:
                        word_text = ", ".join([f"**{word}** ({count})" for word, count in top_words[:10]])
                        st.markdown(word_text)
                    
                    st.success(f"✅ Analyzed and added to memory!")
                    
                    # Clean up temp file
                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)


# MODE: Create File
elif mode == "Create File":
    st.header("✨ Generate File")
    
    filename = st.text_input("Output filename:", placeholder="example.md")
    prompt = st.text_area(
        "Describe what you want to create:",
        placeholder="Create a Python script that...",
        height=100
    )
    
    if st.button("Generate", use_container_width=True, type="primary"):
        if filename and prompt:
            with st.spinner("Generating..."):
                try:
                    content = create_file_from_prompt(filename, prompt)
                    st.success(f"✅ Created: {filename}")
                    
                    # Show the generated content
                    with open(filename, "r") as f:
                        file_content = f.read()
                    st.code(file_content, language="python" if filename.endswith(".py") else "markdown")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a filename and description")


# MODE: Memory
elif mode == "Memory":
    st.header("💾 Memory & History")
    
    memory_data = show_memory()
    
    st.subheader("Recent Messages")
    messages = memory_data.get("messages", [])
    if messages:
        for msg in messages[-20:]:
            role = msg.get("role", "unknown").upper()
            text = msg.get("text", "")[:200]  # Truncate
            st.caption(f"**{role}:** {text}...")
    else:
        st.info("No messages yet")
    
    st.subheader("Analyzed Files")
    analyses = memory_data.get("analyses", [])
    if analyses:
        for a in analyses[-10:]:
            path = a.get("path", "unknown")
            analysis = a.get("analysis", {})
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📄 {path}")
            with col2:
                st.caption(f"{analysis.get('words', 0)} words")
    else:
        st.info("No files analyzed yet")
    
    st.subheader("Raw Memory (JSON)")
    st.json(memory_data)


# MODE: Help
elif mode == "Help":
    st.header("❓ Help & Guide")
    
    st.markdown("""
    ### How to use the Assistant
    
    **💬 Chat**
    - Ask questions and the assistant responds
    - It remembers recent conversation history
    - Better responses if you analyze files first
    
    **📄 Analyze File**
    - Upload a PDF, text, code, or markdown file
    - The assistant reads it and extracts key information
    - Analyzed files are added to memory for context
    
    **✨ Generate File**
    - Ask the assistant to create new files
    - Describe what you want and it generates it
    - Useful for creating code templates, documentation, etc.
    
    **💾 Memory**
    - View all messages and analyzed files
    - See the full conversation history
    - Raw JSON export of all data
    
    ### Tips
    
    1. **Analyze first** - Upload relevant files before asking questions
    2. **Be specific** - More detailed prompts = better responses
    3. **Use context** - Reference analyzed files in your questions
    4. **Iterate** - Ask follow-up questions to refine responses
    
    ### Available LLM Backends
    
    - **GPT4All** (default) - Local, automatic, no setup
    - **Ollama** - More control, custom models
    - **OpenRouter** - Powerful cloud models with API key
    
    First run may take a few minutes as it downloads the model.
    """)


st.divider()
st.caption("🚀 Spring-2026 Assistant | Powered by Streamlit + LLM")
