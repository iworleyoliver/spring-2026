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
    analyses = mem.recent_analyses(3)
    msgs = mem.recent_messages(6)
    parts: List[str] = []
    if analyses:
        parts.append("Recent analyses:")
        for a in analyses:
            p = a.get("path")
            an = a.get("analysis", {})
            top = ", ".join(w for w, _ in an.get("top_words", []))
            parts.append(f"- {p}: pages={an.get('n_pages', 'N/A')} words={an.get('words')} top=[{top}]")
    if msgs:
        parts.append("Recent messages:")
        for m in msgs:
            parts.append(f"- {m.get('role')}: {m.get('text')}")
    return "\n".join(parts)


def respond(prompt: str) -> str:
    """Generate a simple response using recent context and the prompt.
    This will use an external LLM (OpenAI) when `OPENAI_API_KEY` is set; otherwise
    it falls back to the local deterministic generator.
    """
    mem.add_message("user", prompt)
    # Try LLM first (if available)
    ctx = _build_context_summary()
    if llm_generate is not None:
        try:
            # Pass context and prompt to the LLM wrapper
            llm_resp = llm_generate(prompt, context=ctx)
            mem.add_message("assistant", llm_resp)
            return llm_resp
        except Exception:
            # Fall through to deterministic generator
            pass

    # Deterministic fallback generator (existing behavior)
    lines: List[str] = []
    lines.append("Assistant response")
    lines.append("Prompt: " + prompt)
    if ctx:
        lines.append("\nContext:\n" + ctx)
    lines.append("\nGenerated content:\n")
    # If the user asked to "create" or "generate" include a small scaffold
    p_low = prompt.lower()
    if "create" in p_low or "generate" in p_low or "make" in p_low:
        # create a short scaffold using top words if available
        analyses = mem.recent_analyses(1)
        top_words = [w for w, _ in analyses[-1]["analysis"].get("top_words", [])] if analyses else []
        title = prompt.strip().capitalize()
        lines.append(f"# {title}\n")
        if top_words:
            lines.append("Keywords: " + ", ".join(top_words[:8]) + "\n")
        lines.append("Summary:\nThis file was generated based on the provided prompt and recent file analyses.\n")
        lines.append("Content:\n")
        # produce a few lines that use top words where possible
        for i in range(3):
            if top_words:
                w = top_words[i % len(top_words)]
                lines.append(f"- Discuss {w} and how it relates to {title}.")
            else:
                lines.append(f"- Discuss aspect {i+1} related to the prompt.")
    else:
        lines.append("I understood your request. I can create files or summarize analyzed documents. Try 'create' or 'generate' in your prompt to make a file.")

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


def show_memory() -> Dict[str, Any]:
    return mem.show()
