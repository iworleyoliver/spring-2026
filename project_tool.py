"""
CLI for creating project skeletons and analyzing files/PDFs.
Usage examples:
  python3 project_tool.py create-project myproj
  python3 project_tool.py analyze-file README.md
"""
import argparse
import os
import json
from analyzer import analyze_file


def create_project(name: str, with_venv: bool = False) -> None:
    os.makedirs(name, exist_ok=True)
    os.makedirs(os.path.join(name, "src"), exist_ok=True)
    os.makedirs(os.path.join(name, "data"), exist_ok=True)
    with open(os.path.join(name, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {name}\n\nGenerated project scaffold.\n")
    with open(os.path.join(name, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("# Add your dependencies here\n")
    print(f"Created project skeleton in {name}")
    if with_venv:
        print("(Note) To create a virtualenv, run: python3 -m venv {name}/.venv")


def analyze(path: str, top_n: int = 10) -> None:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    try:
        stats = analyze_file(path, top_n=top_n)
    except Exception as e:
        print(f"Error while analyzing: {e}")
        return
    # Print results as JSON for easy scripting
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Project helper: scaffold and analyze files/PDFs")
    sub = parser.add_subparsers(dest="cmd")

    p_create = sub.add_parser("create-project", help="Create a project skeleton")
    p_create.add_argument("name")
    p_create.add_argument("--venv", action="store_true", help="Print venv suggestion")

    p_analyze = sub.add_parser("analyze-file", help="Analyze a file or PDF and print stats as JSON")
    p_analyze.add_argument("path")
    p_analyze.add_argument("--top", type=int, default=10, help="Top N words to show")

    p_a_add = sub.add_parser("assistant-analyze-add", help="Analyze a file and add analysis to assistant memory")
    p_a_add.add_argument("path")
    p_a_add.add_argument("--top", type=int, default=10, help="Top N words to store")

    p_chat = sub.add_parser("assistant-chat", help="Send a prompt to the local assistant (stateful)")
    p_chat.add_argument("prompt", nargs='+', help="Prompt to send to the assistant")

    p_create = sub.add_parser("assistant-create-file", help="Ask assistant to create a file from a prompt")
    p_create.add_argument("path", help="Output file path to create")
    p_create.add_argument("prompt", nargs='+', help="Prompt to use when creating the file")

    p_mem = sub.add_parser("assistant-memory-show", help="Show assistant memory (messages and analyses)")

    args = parser.parse_args()
    if args.cmd == "create-project":
        create_project(args.name, with_venv=args.venv)
    elif args.cmd == "analyze-file":
        analyze(args.path, top_n=args.top)
    elif args.cmd == "assistant-analyze-add":
        from assistant import analyze_and_remember
        res = analyze_and_remember(args.path, top_n=args.top)
        import json
        print(json.dumps(res, indent=2))
    elif args.cmd == "assistant-chat":
        from assistant import respond
        prompt = " ".join(args.prompt)
        print(respond(prompt))
    elif args.cmd == "assistant-create-file":
        from assistant import create_file_from_prompt
        prompt = " ".join(args.prompt)
        out = create_file_from_prompt(args.path, prompt)
        print(f"Created {out}")
    elif args.cmd == "assistant-memory-show":
        from assistant import show_memory
        import json
        print(json.dumps(show_memory(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
