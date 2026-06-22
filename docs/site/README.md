# PG-MAP project page

A single-file static site that serves as the public project page for the PG-MAP preprint (Ruolan Sun, Pawel Polak · Stony Brook University), currently under review at NeurIPS 2026. Self-contained: no JavaScript, no CDN, no external fonts.

## Contents

```
docs/site/
├── index.html      # main page (abstract, method, results table, paired gallery)
├── style.css       # all styles in a single file, no external dependencies
└── images/         # 16 web-optimized JPEGs (~2.4 MB total)
```

## How to serve

### Option A — host on GitHub Pages

In the repository **Settings → Pages**, set the source to **`main` branch, `/docs` folder**, then point any browser at `https://<owner>.github.io/PG-MAP/site/`.

GitHub Pages will pick up `docs/site/index.html` automatically. For the public repo this serves at `https://sophialanlan.github.io/PG-MAP/site/`.

### Option B — package as a ZIP

```bash
# From the repo root
cd docs/site
zip -r ../pgmap_site.zip .
```

The resulting `pgmap_site.zip` (~2.4 MB) can be shared directly or attached to a release. Anyone extracts it and double-clicks `index.html` — the page is fully self-contained (no JS, no CDN, no external fonts).

### Option C — local preview

```bash
cd docs/site
python -m http.server 8000
# then open http://localhost:8000
```

## Page metadata

- Authors: **Ruolan Sun, Pawel Polak · Stony Brook University** (set in the `.authors` line of `index.html`).
- Venue: the header notes the paper is a preprint, under review at NeurIPS 2026.
- Public repository: <https://github.com/sophialanlan/PG-MAP>.

To update the author line or the venue note, edit the `<p class="authors">` line and the note paragraph near the top of `index.html`.

## Updating the gallery

The paired images live in `images/`. To swap in a different example pair:

1. Drop the two PNGs (or JPGs) into `images/`.
2. Convert/compress (max 1024 px on long side, JPEG quality 88 keeps total page size under 3 MB):
   ```python
   from PIL import Image
   img = Image.open("new_image.png").convert("RGB")
   img.thumbnail((1024, 1024), Image.LANCZOS)
   img.save("docs/site/images/new_image.jpg", "JPEG", quality=88, optimize=True, progressive=True)
   ```
3. Add a new `<article class="pair">` block in `index.html` (copy any existing one as a template).

## Browser support

Plain HTML 5 + CSS Grid. Tested mentally against current Chrome, Firefox, Safari. No transpilation, no build step, no JavaScript.
