# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tech Stack

Personal data science blog built with **Hugo** (PaperMod theme) deployed to GitHub Pages. Posts are authored as Jupyter notebooks and converted to Hugo markdown via **Quarto**. Math is rendered with KaTeX. The site supports English (primary) and Russian.

## Build Commands

```bash
hugo server -D      # local dev server at localhost:1313, includes drafts
hugo                # production build → /docs (GitHub Pages source)
```

## Notebook-to-Post Workflow

1. Create `content/posts/<post-name>/` and place a `.ipynb` notebook inside.
2. Convert with Quarto (PowerShell — `python` defaults to 3.14, use explicit 3.11 path):
   ```powershell
   $env:QUARTO_PYTHON = "C:\Users\podlo\AppData\Local\Programs\Python\Python311\python.exe"
   & "C:\Program Files\Quarto\bin\quarto.exe" render content/posts/<post-name>/Notebook.ipynb
   ```
3. Move the generated `.md` one level up:
   ```bash
   mv content/posts/<post-name>/Notebook.md content/posts/<post-name>.md
   ```
4. If widget `<script>` tags didn't render, comment them out in the markdown.
5. Set `draft = false` in the front matter when ready to publish, then run `hugo` and push.

### Notebook front matter (YAML)
```yaml
---
title: "Post Title"
format:
    hugo-md:
        output-file: "post-name.md"
        html-math-method: katex
        code-fold: true
jupyter: python3
execute:
    enabled: false
    cache: true
---
```

## Key Conventions

- **File naming:** The `.md` filename must exactly match the post folder name — mismatches break chart paths.
- **KaTeX:** Avoid `\text{}` in LaTeX formulas; use plain text instead. KaTeX does not render `\text` correctly.
- **Multi-language:** Russian translations use a `.ru.md` suffix (e.g., `post-name.ru.md`).
- **Deployment:** Push to `master`; GitHub Pages publishes from `/docs` automatically.
