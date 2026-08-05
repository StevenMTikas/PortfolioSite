# steventikas.online

Steven Tikas's personal portfolio site: a static single-page site with an AI chat widget ("MeBot") that answers visitor questions about him.

Live at [steventikas.online](https://steventikas.online), hosted on Vercel.

## Stack

- **Site**: plain HTML/CSS/JS in [index.html](index.html) and [styles.css](styles.css), styled with the Tailwind CDN build (no build step, no framework).
- **Chatbot backend**: a single Vercel serverless function at [api/ask.js](api/ask.js), Node.js, zero npm dependencies (uses the runtime's built-in `fetch`).
- **Hosting**: [Vercel](https://vercel.com), auto-deployed from the `main` branch via its GitHub integration — pushing to `main` is the deploy. There is no manual deploy step and no CLI needed.

## Project structure

```
index.html                 The entire site: markup, Tailwind classes, and all page JS (inline <script> blocks)
styles.css                 Small amount of custom CSS not covered by Tailwind utility classes
api/
  ask.js                   Serverless function backing the chat widget (POST /api/ask)
  _context.js              Bio + resume text fed to the chatbot as context (plain JS string constants)
assets/
  Steven_Tikas_AI_Solutions_Engineer.pdf   Resume PDF linked from the hero section
  MeBot/
    bio.txt                Source text _context.js's BIO_TEXT was drawn from
    LinkedIn_Profile.pdf   Not currently used by anything (see "Loose ends" below)
chatbot/index.html          Standalone full-page chat demo, orphaned (see "Loose ends" below)
images/cover_image.jpg      Open Graph / social preview image
vercel.json                 Vercel routing config (SPA-style rewrite to index.html)
.vercelignore                Files excluded from the Vercel deployment
```

## Running locally

There's no build step for the site itself — open `index.html` directly in a browser, or serve the folder with anything static (e.g. `npx serve`). The chat widget won't work this way, though, since it calls `/api/ask`, which only exists once deployed to Vercel (or run through the Vercel CLI's `vercel dev`).

To test `api/ask.js` in isolation without deploying, run it directly in Node with a mock request — see the shape used in this repo's development history, or just:

```js
require('dotenv').config();
const handler = require('./api/ask.js');
handler(
  { method: 'POST', body: { message: 'What does Steven do?' } },
  { status(c) { this._s = c; return this; }, json(o) { console.log(this._s, o); } }
);
```

This needs `OPENAI_API_KEY` available in the environment (e.g. a local `.env` file, gitignored).

## The chatbot (`api/ask.js`)

The widget in `index.html` POSTs `{ message }` to `/api/ask` and renders back `{ reply }`.

The knowledge base is small (a bio and a resume, a few KB of text total), so there's no vector store, embeddings, or RAG — the full bio and resume text from `api/_context.js` are just stuffed directly into the prompt sent to OpenAI (`gpt-4o-mini`). A short-lived in-memory cache (10 min) holds a live fetch of Steven's public GitHub repos, so the bot can also answer questions about current projects. The `_context.js` texts are treated as the primary source of truth; GitHub is secondary; nothing else is consulted.

This intentionally replaced an earlier version of the chatbot that ran on Render as a Python FastAPI service using LangChain + FAISS for real RAG. That backend kept getting suspended by Render's free tier from inactivity, which silently broke the widget for extended periods. The current version has no server to suspend and no cold-start vector store to rebuild.

### Required environment variables (set in Vercel, not committed)

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Chat completions. Without it, `/api/ask` returns a 503 and the widget shows a connection error. |
| `GITHUB_TOKEN` | No | Raises the GitHub API rate limit for the live repo-fetch (60/hr unauthenticated → 5,000/hr with a token). |

Set these under the Vercel project's **Settings → Environment Variables**. A local `.env` file (gitignored) is only needed for running/testing the function outside Vercel.

## Updating the chatbot's knowledge

`api/_context.js` is the single source of truth the bot reads from — there's no build step pulling from `bio.txt` automatically. To change what the bot knows, edit the `BIO_TEXT` / `RESUME_TEXT` constants in `api/_context.js` directly and redeploy (push to `main`). `assets/MeBot/bio.txt` is kept as the original source text for reference but isn't read at request time.

## Loose ends / things worth deciding on

- **`chatbot/index.html`** — a standalone full-page version of the chat UI, originally served by the old Render backend's root route. It's excluded from the Vercel deploy via `.vercelignore` and isn't linked from anywhere live. The "MeBot" project card on the site itself has its demo link disabled (`url: null` in `index.html`) pointing at this. Either wire it up to the new `/api/ask` endpoint and re-enable the card, or remove it.
- **`assets/MeBot/LinkedIn_Profile.pdf`** — not read by anything anymore (the old Python backend used to ingest all PDFs in that folder; the new chatbot doesn't). Safe to delete or fold into `RESUME_TEXT` in `_context.js` if there's information in it worth surfacing.
- **Other disabled project links** — YouTube Product Crew, NFL Picker, and AI News Aggregator all have `url: null` in `index.html`'s project data, with a comment noting their Render-hosted demos were suspended as of 2026-08-04. Same fate as the old chatbot backend; worth revisiting if any of those are still worth demoing live.
