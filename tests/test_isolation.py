import ast
from pathlib import Path


def test_no_mlx_audio_dependency():
    src_dir = Path(__file__).resolve().parent.parent / "src"
    python_files = list(src_dir.rglob("*.py"))
    assert len(python_files) > 0, "No python files found in src"

    for py_file in python_files:
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mlx_audio" not in alias.name, (
                        f"Forbidden import '{alias.name}' in {py_file}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "mlx_audio" not in module, (
                    f"Forbidden from-import '{module}' in {py_file}"
                )
