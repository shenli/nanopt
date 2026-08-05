from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_strict_docs_build_emits_mathjax_ready_page(tmp_path: Path, project_root: Path) -> None:
    output = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(output)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    html = (output / "foundations/mathjax-smoke/index.html").read_text()
    mathjax_config = (output / "javascripts/mathjax.js").read_text()
    assert 'class="arithmatex"' in html
    assert "\\(z_i\\)" in html
    assert "\\[" in html
    assert "tex-mml-chtml.js" in html
    assert 'inlineMath: [["\\\\(", "\\\\)"]]' in mathjax_config
    assert 'displayMath: [["\\\\[", "\\\\]"]]' in mathjax_config
    assert 'processHtmlClass: "arithmatex"' in mathjax_config
