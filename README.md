### Quick Start

 1. Create a new folder in `content/posts/` and add new `.ipynb` article there
 2. Execute the notebook either way:
	- manually
	- pass `--execute` flag to `quarto preview` or `quarto render`
	- set `enabled` flag in your notebook front matter
    For the last two scenarious don't forget to set env var to use the same Python kernel as used in your notebook, in my case `QUARTO_PYTHON=python311`
 3. Run `quarto preview content/posts/<your-folder>/<your-notebook>.ipynb`
 4. Run `hugo server -D` to test the site locally including draft posts
 5. Comment the header <scripts> in you `.md` file if it wasn't rendered correctly
 6. Move `.md` file one level up to `content/posts/`
 6. Deploy to production with simple `hugo` command

### Backlog

#### Done
 - Add RU language
 - Enable SEO indexing - [Google Search Console](https://search.google.com/search-console?resource_id=https%3A%2F%2Fnpodlozhniy.github.io%2F)
 - Turn on Google Analytics - [GA Report](https://analytics.google.com/analytics/web/#/p410388205/reports/intelligenthome?params=_u..nav%3Dmaui)
 - Set up RSS
 - Make comments availabile
 - Make comments theme consistent
 - Add favicon

#### To Do
 - Fix search engine (still unclear, works locally)
 - Fix surname misspelling in domain name