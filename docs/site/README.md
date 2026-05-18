# Anonymous project page

A single-file static site presenting the paper to reviewers. **No author names, no affiliations, no identifying links** — safe to upload as supplementary material for double-blind review.

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

GitHub Pages will pick up `docs/site/index.html` automatically.

> Caveat: enabling GitHub Pages on a repo named `<owner>/PG-MAP` reveals the GitHub owner. For a strictly-anonymous reviewer link, prefer Option B during the review period.

### Option B — upload as supplementary ZIP

```bash
# From the repo root
cd docs/site
zip -r ../pgmap_anon_site.zip .
```

The resulting `pgmap_anon_site.zip` (~2.4 MB) can be uploaded directly to OpenReview as anonymous supplementary material. Reviewers extract and double-click `index.html` — the page is fully self-contained (no JS, no CDN, no external fonts).

### Option C — local preview

```bash
cd docs/site
python -m http.server 8000
# then open http://localhost:8000
```

## What's anonymized

- No author names appear anywhere in the HTML.
- No affiliations or grant numbers.
- No GitHub usernames or external links other than the anchor links inside the page.
- The footer says simply "Anonymous · supplementary material for NeurIPS 2026 submission".

If you need to update the page after acceptance to add author info, edit `index.html` and replace the line `<p class="authors">Anonymous authors</p>` plus the anonymized note in `.anonymous-note`.

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
