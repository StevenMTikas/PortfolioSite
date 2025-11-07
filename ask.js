// /api/ask.js
export default async function handler(req, res) {
  // Only allow POST
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { message } = req.body || {};

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Missing 'message' string in body." });
    }

    // call the model
    // NOTE: this uses the OpenAI-style API schema.
    // If you're using something else (like Groq or OpenRouter)
    // you'll update baseURL/model below.
    const apiKey = process.env.OPENAI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({ error: "Server misconfigured: no API key." });
    }

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini", // cheap & fast; swap if you want
        messages: [
          {
            role: "system",
            content:
              "You are an AI assistant that speaks on behalf of the developer. Be helpful, confident, and specific about what they can build (automation, AI content tools, YouTube workflow, early learning tutor, etc.). Keep answers under 200 words unless you're asked to go deep.",
          },
          {
            role: "user",
            content: message,
          },
        ],
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("OpenAI API error:", errText);
      return res.status(500).json({ error: "Upstream model error." });
    }

    const data = await response.json();

    // Extract assistant message text safely
    const reply =
      data?.choices?.[0]?.message?.content?.trim() ||
      "I wasn't able to generate a reply.";

    return res.status(200).json({ reply });
  } catch (err) {
    console.error("Server exception:", err);
    return res.status(500).json({ error: "Unexpected server error." });
  }
}
