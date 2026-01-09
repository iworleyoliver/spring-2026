"""
A small, local, stateful assistant that can:
- remember conversation history and analyses
- analyze files (calls analyzer.analyze_file)
- generate simple file contents from prompts and analysis context
- create files and store the action in memory

Memory persisted to `.assistant_memory.json` in the repo root.
"""
import json
import os
from typing import Any, Dict, List, Optional
from analyzer import analyze_file
try:
    from llm_backend import generate as llm_generate
except Exception:
    llm_generate = None  # type: ignore

MEM_PATH = ".assistant_memory.json"


class Memory:
    def __init__(self, path: str = MEM_PATH):
        self.path = path
        self._data: Dict[str, Any] = {"messages": [], "analyses": []}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {"messages": [], "analyses": []}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def add_message(self, role: str, text: str):
        self._data.setdefault("messages", []).append({"role": role, "text": text})
        self._save()

    def add_analysis(self, path: str, analysis: Dict[str, Any]):
        self._data.setdefault("analyses", []).append({"path": path, "analysis": analysis})
        self._save()

    def recent_analyses(self, n: int = 3):
        return list(self._data.get("analyses", []))[-n:]

    def recent_messages(self, n: int = 10):
        return list(self._data.get("messages", []))[-n:]

    def show(self):
        return self._data


mem = Memory()


def analyze_and_remember(path: str, top_n: int = 10) -> Dict[str, Any]:
    analysis = analyze_file(path, top_n=top_n)
    mem.add_analysis(path, analysis)
    mem.add_message("system", f"Analyzed file {path}: {analysis}")
    return analysis


def _build_context_summary() -> str:
    analyses = mem.recent_analyses(5)
    msgs = mem.recent_messages(10)
    parts: List[str] = []
    
    # Include analyzed files with better formatting
    if analyses:
        parts.append("## Recent File Analyses")
        for a in analyses:
            p = a.get("path")
            an = a.get("analysis", {})
            top = ", ".join(w for w, _ in an.get("top_words", [])[:5])
            n_pages = an.get("n_pages", "N/A")
            page_info = f" ({n_pages} pages)" if n_pages != "N/A" else ""
            parts.append(f"- **{p}**{page_info}: {an.get('words', 0)} words, topics: {top}")
    
    # Include conversation flow with better context
    if msgs:
        parts.append("\n## Conversation Flow")
        # Group messages into exchanges
        recent_pairs = []
        for i in range(len(msgs)-1, -1, -2):
            if i >= 0:
                recent_pairs.insert(0, msgs[i])
            if i-1 >= 0:
                recent_pairs.insert(0, msgs[i-1])
        
        for msg in recent_pairs[-8:]:  # Last 4 exchanges
            role = msg.get('role', 'unknown').capitalize()
            text = msg.get('text', '')[:150]
            parts.append(f"- **{role}**: {text}...")
    
    # Add summary of conversation topics
    if msgs:
        all_text = " ".join([m.get('text', '') for m in msgs])
        if "code" in all_text.lower():
            parts.append("\n**Focus**: Code-related discussions")
        if "analyze" in all_text.lower() or "analyze" in all_text.lower():
            parts.append("**Focus**: File analysis and documentation")
    
    return "\n".join(parts) if parts else "No context available yet. Analyze some files first with 'Analyze File' mode."


def respond(prompt: str) -> str:
    """Generate a response using recent context and the prompt.
    Uses LLM when available (Ollama or OpenRouter), with intelligent fallback.
    """
    mem.add_message("user", prompt)
    ctx = _build_context_summary()
    
    # Try LLM first (if available)
    if llm_generate is not None:
        try:
            # Build a better system prompt for the LLM
            system_context = (
                "You are a helpful coding assistant. You have access to recent file analyses "
                "and conversation history. Use this context to provide relevant, concise answers. "
                "Be direct and practical in your responses."
            )
            full_context = f"{system_context}\n\n{ctx}" if ctx != "No context available yet." else system_context
            llm_resp = llm_generate(prompt, context=full_context)
            mem.add_message("assistant", llm_resp)
            return llm_resp
        except Exception as e:
            # Log failure and fall through to intelligent fallback
            pass

    # Intelligent fallback generator
    return _fallback_response(prompt, ctx)


def _fallback_response(prompt: str, context: str) -> str:
    """Generate a thoughtful fallback response when LLM is unavailable."""
    p_low = prompt.lower()
    lines: List[str] = []
    
    # Analyze intent from prompt
    if any(word in p_low for word in ["summarize", "summary", "overview"]):
        lines.append("## Summary based on recent analyses:\n")
        analyses = mem.recent_analyses(3)
        if analyses:
            for a in analyses:
                path = a.get("path", "unknown")
                analysis = a.get("analysis", {})
                words = analysis.get("words", 0)
                top_words = ", ".join(w for w, _ in analysis.get("top_words", [])[:3])
                lines.append(f"- **{path}**: {words} words covering {top_words}")
        else:
            lines.append("No files have been analyzed yet. Use 'assistant-analyze-add <file>' to analyze files.")
    
    elif any(word in p_low for word in ["create", "generate", "write", "make"]):
        analyses = mem.recent_analyses(1)
        top_words = [w for w, _ in analyses[-1]["analysis"].get("top_words", [])] if analyses else []
        title = prompt.replace("create", "").replace("generate", "").replace("write", "").strip().capitalize() or "Untitled"
        
        lines.append(f"# {title}\n")
        lines.append(f"Generated based on: {' '.join(top_words[:3]) if top_words else 'prompt'}\n")
        lines.append("## Overview")
        lines.append("This document was generated in response to your request.\n")
        
        lines.append("## Key Points")
        if top_words:
            for i, word in enumerate(top_words[:4], 1):
                lines.append(f"{i}. **{word.capitalize()}**: [Add details about {word}]")
        else:
            lines.append("1. Main point\n2. Supporting detail\n3. Conclusion")
        
        lines.append("\n## Conclusion\nThis generated content provides a starting point for your needs.")
    
    elif any(word in p_low for word in ["help", "how", "what", "explain", "?"]):
        lines.append("### Available Commands:\n")
        lines.append("- `assistant-analyze-add <file>` - Analyze a file and add to memory")
        lines.append("- `assistant-chat \"<prompt>\"` - Chat with the assistant")
        lines.append("- `assistant-create-file <path> \"<prompt>\"` - Generate a file")
        lines.append("- `assistant-memory-show` - View conversation history\n")
        lines.append("Try asking me to summarize your files, or create something based on them!")
    
    else:
        lines.append(f"**Understood**: {prompt}\n")
        analyses = mem.recent_analyses(1)
        if analyses:
            top_words = [w for w, _ in analyses[-1]["analysis"].get("top_words", [])]
            lines.append(f"Based on recent analysis, relevant terms: {', '.join(top_words[:3])}")
        else:
            lines.append("I'm ready to help! Analyze some files first with `assistant-analyze-add` to give me context.")

    resp = "\n".join(lines)
    mem.add_message("assistant", resp)
    return resp


def create_file_from_prompt(path: str, prompt: str) -> str:
    content = respond(prompt)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    mem.add_message("system", f"Created file {path} from prompt")
    return path


def trim_memory(max_messages: int = 50, max_analyses: int = 20) -> None:
    """Trim memory to prevent unbounded growth."""
    data = mem._data
    if len(data.get("messages", [])) > max_messages:
        data["messages"] = data["messages"][-max_messages:]
    if len(data.get("analyses", [])) > max_analyses:
        data["analyses"] = data["analyses"][-max_analyses:]
    mem._save()


def show_memory() -> Dict[str, Any]:
    return mem.show()
