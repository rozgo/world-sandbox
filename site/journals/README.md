# Journals

Each World Sandbox project has a journal: its story, its films, its figures and its measured results.
The home page (`hub/`) links them all. The layout, components and rules follow the MuJoCo Sandbox's
journals (`rozgo/mujoco-sandbox`, `site/journals/README.md`), with two additions: an interactive 3D
viewer (`site/viewer/`) and inline diagrams.

```sh
uv run --locked python scripts/build_journals.py                  # every journal and the home page
uv run --locked python scripts/build_journals.py synthetic_slam   # one journal
python3 -m http.server 8792 --bind 127.0.0.1 --directory build/journals
scripts/publish_pages.sh build/journals/synthetic_slam synthetic_slam
scripts/publish_pages.sh build/journals/hub .                     # the home page, at the site root
```

## A journal

`site/journals/<id>/journal.json` holds the title, heading, eyebrow, lede, description, hero film,
metric tiles, scope, footer, home-page card, media list and theme (fields as in the MuJoCo Sandbox).
An optional `viewer` entry, `{"data": <folder with scene.json and points.bin>, "video": <mp4>}`, copies
`site/viewer/` into the journal's `viewer/` with that run's data and a web copy of its video.

`site/journals/<id>/content.html` holds the chapters, each
`<section id="slug"><h2>N · Title</h2><p class="when">date · context</p> ... </section>`. Media are
`media/<key>.mp4` (with `media/<key>_poster.webp`) or `media/<key>.webp`. Besides the MuJoCo Sandbox
components there are `<figure class="embed"><iframe src="viewer/">…` and `<figure class="diagram"><svg>…`.

## Rules

- Every number comes from a file in the repository; say what was measured and under what conditions.
  Keep failures on record. Say which inputs are generated and which parts are learned or programmed.
- Plain, concise, public-safe: no machine connection details, credentials or private names.
- Keep a built journal under about 70 MB.

## Palettes

| Journal | Scheme | Character | Key colours |
| --- | --- | --- | --- |
| Synthetic SLAM | dark | scrapyard: graphite, rust, oxidised teal, warning yellow | `#0f1113` `#e9e6e1` `#e0703a` `#4fb3a9` `#e8b931` |
