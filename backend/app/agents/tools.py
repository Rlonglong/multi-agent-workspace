from langchain_core.tools import tool
import os
import subprocess
import tempfile
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORKSPACE_DIR = os.path.abspath(
    os.getenv("AGENT_WORKSPACE_DIR", os.path.join(REPO_ROOT, "agent_workspace"))
)
os.makedirs(WORKSPACE_DIR, exist_ok=True)


def resolve_workspace_path(filepath: str) -> str:
    normalized = os.path.normpath(filepath).lstrip(os.sep)
    full_path = os.path.abspath(os.path.join(WORKSPACE_DIR, normalized))
    if full_path != WORKSPACE_DIR and not full_path.startswith(WORKSPACE_DIR + os.sep):
        raise ValueError("Refusing to write outside workspace root.")
    return full_path


def cleanup_workspace_files(filepaths: list[str]) -> list[str]:
    removed: list[str] = []
    for filepath in filepaths or []:
        try:
            full_path = resolve_workspace_path(filepath)
        except Exception:
            continue
        if os.path.isfile(full_path):
            os.remove(full_path)
            removed.append(filepath)
            parent = os.path.dirname(full_path)
            while parent.startswith(WORKSPACE_DIR) and parent != WORKSPACE_DIR:
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
            removed.append(filepath)
    return removed


def reset_workspace_dir() -> None:
    for entry in os.listdir(WORKSPACE_DIR):
        full_path = os.path.join(WORKSPACE_DIR, entry)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
        else:
            try:
                os.remove(full_path)
            except FileNotFoundError:
                pass

@tool
def write_code_file(filepath: str, content: str) -> str:
    """
    Save generated code to the isolated agent workspace directory.
    This workspace is separate from the host app repo.
    """
    full_path = resolve_workspace_path(filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Successfully wrote file: {os.path.relpath(full_path, WORKSPACE_DIR)}"

@tool
def read_code_file(filepath: str) -> str:
    """
    Read an existing file from the isolated agent workspace.
    """
    full_path = resolve_workspace_path(filepath)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File {filepath} not found."

@tool
def execute_playwright_qa(script_code: str) -> str:
    """
    QA Agents ONLY: Execute a Python Playwright script to test the UI.
    Your script must import sync_playwright and print assertions.
    Ensure you point the browser to localhost:3000 where the Next.js app is hosted.
    Example:
    ```python
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto('http://localhost:3000')
        print(f"Title: {page.title()}")
        browser.close()
    ```
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_code)
        temp_name = f.name
        
    try:
        result = subprocess.run(['python3', temp_name], capture_output=True, text=True, timeout=30)
        output = f"STDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}"
        if result.returncode == 0:
            return f"✅ Playwright Test Passed!\\n{output}"
        else:
            return f"❌ Playwright Test Failed (Code {result.returncode}):\\n{output}"
    except subprocess.TimeoutExpired:
        return "❌ Playwright Execution Timeout (exceeded 30 seconds)."
    except Exception as e:
        return f"❌ Execution Error: {str(e)}"
    finally:
        os.unlink(temp_name)
        
    return "❌ Playwright Test completed with unknown state."
