"""Project startup hooks.

Python imports this module automatically when the repository root is on sys.path.
It applies narrow runtime fixes for the TypeFly LLM controller.
"""

try:
    from controller.llm_controller_runtime_fixes import apply_runtime_fixes
except Exception:
    apply_runtime_fixes = None

if apply_runtime_fixes is not None:
    try:
        apply_runtime_fixes()
    except Exception:
        pass
