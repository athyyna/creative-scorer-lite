# Creative Scorer Lite

**Image compliance and AI performance scoring for Meta, TikTok, Google PMax, and YouTube.**

Built by Manus for Athena · Media Strategy

---

## What it does

Upload 1–5 image assets and get instant feedback on:

- **Compliance checks** — format, file size, resolution, aspect ratio, text density, and safe zones per platform
- **AI performance score** (0–100) across 5 weighted dimensions: Hook Strength, Message Clarity, Mobile Fit, Audience Fit, and Native Feel
- **Verdict** — Strong / Run with Revisions / Rework
- **Batch summary table** — ranked by performance score for multi-asset uploads
- **Compliance Legend** — full reference guide for every check and threshold

## Platforms supported

Meta · TikTok · Google PMax · YouTube

## Accepted file formats

JPG · PNG · WebP (up to 5 assets per batch)

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select this repo, branch `main`, file `app.py`
4. Click Deploy

> **Note:** AI scoring requires an OpenAI API key. Add it as a Streamlit secret (`OPENAI_API_KEY`) or paste it in the sidebar at runtime. Compliance checks run without it.

---

## Lite vs Full version

This is the **Lite** version — image-only, no database, no auth, no video support.

The full version (React + Node.js + MySQL) is deployed at [creativesco-fqvs2fyw.manus.space](https://creativesco-fqvs2fyw.manus.space) and includes video compliance checks, scoring history, user auth, and a persistent database.

---

## Compliance specs (2026)

| Check | Meta | TikTok | Google PMax | YouTube |
|---|---|---|---|---|
| Max file size | 30 MB | 500 KB | 5 MB | 2 MB |
| Min resolution | 1080×1080 | 720×1280 | 600×314 | 1280×720 |
| Preferred ratio | 1:1, 4:5, 9:16 | 9:16 | 1.91:1 | 16:9 |
| Max text coverage | 20% | 25% | 20% | 20% |

Specs reflect 2026 platform policies. Always verify against official platform help centres before launch.
