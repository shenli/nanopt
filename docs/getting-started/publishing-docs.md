# Publishing the documentation

NanoPT uses MkDocs Material and does not require GitHub Actions to publish its documentation.

## Preview locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

Open `http://127.0.0.1:8000/nanopt/` and inspect the pages before publishing.

## Publish to GitHub Pages

From a clean, reviewed `main` checkout:

```bash
uv run python scripts/lint_formulas.py docs
uv run mkdocs build --strict
uv run mkdocs gh-deploy
```

The last command builds the site locally and pushes only the generated site to the `gh-pages`
branch. In the repository's **Settings → Pages**, choose **Deploy from a branch**, select
`gh-pages` and `/ (root)`, then save.

The configured site URL is:

```text
https://shenli.github.io/nanopt/
```

Repeat the three publish commands whenever reviewed documentation changes. Do not commit `site/`;
it is a local build directory and remains ignored.
