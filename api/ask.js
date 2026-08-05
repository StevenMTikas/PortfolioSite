// Vercel serverless function backing the MeBot chat widget.
// Replaces the old Render-hosted FastAPI/LangChain/FAISS backend: the knowledge
// base here is small enough to stuff directly into the prompt, so no vector
// store is needed, which also avoids Render's free-tier cold-start/suspension.

const { BIO_TEXT, RESUME_TEXT } = require('./_context');

const GITHUB_USERNAME = 'StevenMTikas';

// Repo list is small and changes rarely; a short in-memory cache avoids
// re-fetching GitHub on every request within the same warm function instance.
let githubCache = { text: '', fetchedAt: 0 };
const GITHUB_CACHE_TTL_MS = 10 * 60 * 1000;

async function fetchGithubContext() {
  const now = Date.now();
  if (githubCache.text && now - githubCache.fetchedAt < GITHUB_CACHE_TTL_MS) {
    return githubCache.text;
  }

  const headers = {
    Accept: 'application/vnd.github.v3+json',
    'User-Agent': 'mebot-chat-widget',
  };
  if (process.env.GITHUB_TOKEN) {
    headers.Authorization = `token ${process.env.GITHUB_TOKEN}`;
  }

  try {
    const res = await fetch(
      `https://api.github.com/users/${GITHUB_USERNAME}/repos?type=all&sort=updated&per_page=100`,
      { headers }
    );
    if (!res.ok) return githubCache.text;

    const repos = await res.json();
    const text = repos
      .filter((r) => !r.fork)
      .map((r) => `- ${r.name}: ${r.description || 'No description'} (${r.html_url})`)
      .join('\n');

    githubCache = { text, fetchedAt: now };
    return text;
  } catch {
    return githubCache.text;
  }
}

function buildPrompt({ bio, resume, github, question }) {
  const contextParts = [`PRIMARY SOURCE (biography - OVERRIDING SOURCE OF TRUTH):\n${bio}`];
  if (github) {
    contextParts.push(`SECONDARY SOURCE (GitHub repositories - current projects):\n${github}`);
  }
  contextParts.push(`TERTIARY SOURCE (resume - professional background):\n${resume}`);

  return `You are an AI assistant that answers questions about Steven Tikas based on the provided context. You will try to talk him up when possible without lying or being too over the top about it.

IMPORTANT PRIORITY ORDER:
1. PRIMARY SOURCE (biography) is the OVERRIDING SOURCE OF TRUTH.
2. SECONDARY SOURCE (GitHub repositories) covers current projects.
3. TERTIARY SOURCE (resume) covers professional background.

If sources conflict, prioritize the biography first, then GitHub, then the resume.

Use the following context to answer the question. If you don't know the answer based on the context, say so, but try to be helpful with what you do know.

Context:
${contextParts.join('\n\n---\n\n')}

Question: ${question}

Answer:`;
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ detail: 'Method not allowed' });
    return;
  }

  const message = req.body && typeof req.body.message === 'string' ? req.body.message.trim() : '';
  if (!message) {
    res.status(400).json({ detail: 'Message cannot be empty' });
    return;
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    res.status(503).json({ detail: 'Chatbot is not configured (missing OPENAI_API_KEY).' });
    return;
  }

  try {
    const github = await fetchGithubContext();
    const prompt = buildPrompt({ bio: BIO_TEXT, resume: RESUME_TEXT, github, question: message });

    const completion = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        temperature: 0.7,
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    if (!completion.ok) {
      const errText = await completion.text().catch(() => '');
      console.error('OpenAI error:', completion.status, errText);
      res.status(502).json({ detail: 'Error generating a response.' });
      return;
    }

    const data = await completion.json();
    const reply = data.choices?.[0]?.message?.content?.trim() || "I couldn't generate a response.";

    res.status(200).json({ reply });
  } catch (err) {
    console.error('Error processing question:', err);
    res.status(500).json({ detail: 'Error processing question.' });
  }
};
