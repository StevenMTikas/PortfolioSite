// Static knowledge-base text for the MeBot chatbot.
// Kept as inline constants (rather than read from disk) so the serverless
// function has no filesystem/bundling dependency at request time.

const BIO_TEXT = `Steven Matthew Tikas was born in a little town called Park Ridge which is right outside of Chicago. He was raised by a single mother who worked two jobs all of his life. He learned work ethics and values from her and the other family members in his life.

Steve attended Jane Adams Elementary/Middle School and Palatine High school where he finished with a 3.2 overall GPA. He then went into the military as a Navy Nuclear Machinist Mate and attending A and B schools in Orlando Florida and finished his training in Balston Spa, NY on an old decommissioned reactor. After finishing school he was sent to his first duty station in King's Bay GA where he was stationed on the SSBN-735 Pennsylvania (gold crew). Unfortunately his military career was cut short when he got sick and was medically discharged from the military.

After going home for a few months to deal with medical issues he went to live in Jacksonville, FL with some of his friends. His first job was doing tech support for America Online when they were still popular. He excelled at that job and then they moved him into the billing department where he spent most of his day getting yelled at by customers due to AOL's business practices. After leaving AOL he tried selling cars but the day he started was 9/11/2001 which was not a great day to get into any sales related positions.

Eventually leaving car sales for a temp job at a company called Southeastern Aluminum Products which is a wholesale shower enclosure company. He was eventually promoted to custom order lead and handled processing of over 100 custom orders per day with a less then 2% error rate. After parting from there he went to work for their customers named Lee & Cates Glass which is a customer of Southeastern and actively sought Steve out when they found out he had left Southeastern.

Steve worked at Lee & Cates for a combined 10 years (2 different employment from 2007-2017 and then again from 2019-2021). While there he worked in all facets of the company from the retail side to contract to sales and eventually worked his way into one of the main people for the fabrication facility that was the backbone of the companies business model. He quickly formed strong business relationship with many of their top customers becoming the go-to person when something really needed to get handled quickly and effectively. He was also cross trained on Storefront metal systems. Also while at Lee & Cates Steve completed his BS in Information Technology with a focus on Advanced Business Analystics.

Armed with his new degree he left Lee & Cates on good terms and went to work at his his first true tech support job (AOL was more about script reading then real tech support). He left Bank of America when Lee & Cates called and asked him to come back as a outside salesman. After having a falling out with management Steve decided he would follow a life long dream he had of investing in real estate house flipping and started Father & Sons Home Buyers in July of 2021. Everything was going great and he had successfully flipped his first house when COVID hit and put everyone's life everywhere on hold. During COVID Steve also had a complication of what had forced him out of the military and was unable to work for close to a year.

After recovering he started working at the IT Support Center in a Technical Advisor role. After over 2 years of dedicated work he left to work at Availity Inc which is a middle-man company for insurance claims. Initially starting out as a temp Steve quickly distinguished himself and was the first from his class of over 25 to be asked to come on full time. Due to some complications during the onboarding Steve was forced to go back to the IT Support Center which is where he currently works. He is now a Dedicated Technical Advisor serving as the sole liasion for one of their customer name Med Center Health, a small hospital network in, and around, Bowling Green Kentucky.

More recently, Steve has built a self-directed portfolio of production AI applications: Worxstance (an AI career workspace covering resume tailoring, mock interviews, and networking outreach), SpecAnalyzer (a document-AI pipeline that reads architectural drawing sets for the construction industry), the Tikas Family Digital Home (a self-hosted family platform running local and cloud AI models), and a CrewAI-based YouTube Product Crew multi-agent content pipeline. This work spans OpenAI, Anthropic Claude, Google Gemini, DeepSeek, and local Ollama models, with RAG, structured output, and tool calling.

Steve is happily married to his wife and best friend of almost 20 years now (16 married plus dating), Vashti. They have three beautiful children Malachi, Jordan, and Keyona and one grandson named Tyriq. They all currently live in Jacksonville Florida.

Some interesting facts about me:
I was in the Navy but am a horrible swimmer
I love to laugh and joke around with people
I love the works of JRR Tolkin and Ayn Rand
Favorite movie is V for Vendetta
I have sat on top of a nuclear reactor while painting
I have slept with my feet near a nuclear missile tube
I like everything from Classical to 90/2000s hip-hop and R&B
My mother's name is Carmella
I have 2 sisters: Kristin and Amy
Portfolio Site URL: https://steventikas.online
GitHub: https://github.com/StevenMTikas`;

const RESUME_TEXT = `STEVEN TIKAS — AI SOLUTIONS ENGINEER
Jacksonville, FL | stevenmtikas@gmail.com | linkedin.com/in/steven-m-tikas | steventikas.online

SUMMARY
Builder of production AI applications end to end — LLM integration, multi-agent orchestration, retrieval, and the full-stack products wrapped around them — backed by 20+ years of professional experience across technical support, customer service, and manufacturing. Self-directed portfolio spans five model providers (OpenAI, Anthropic, Google Gemini, DeepSeek, and local inference via Ollama), schema-validated structured output, tool calling, RAG over vector search, and cost-controlled agent pipelines. B.S. in Information Technology.

TECHNICAL SKILLS
AI & LLM: OpenAI, Anthropic Claude, Google Gemini, DeepSeek, local models via Ollama; CrewAI multi-agent orchestration; structured output with response schemas; tool/function calling; RAG with vector embeddings; model routing, caching, and cost control.
Languages: TypeScript, JavaScript, Python, SQL, Shell/PowerShell.
Web & App: React 18/19, Next.js 15, Django, FastAPI, Node.js, Tailwind CSS, React Native (Expo).
Data & Cloud: PostgreSQL, SQLite, Firebase, Microsoft Azure, AWS Textract, Render, Stripe.
Testing & Quality: Vitest, Cypress, pytest, golden-fixture regression harnesses, TypeScript strict mode, ESLint.
Security & Ops: Role-based access control, AES-256-GCM encryption at rest, OAuth 2.0, session authentication, SSO/MFA.

SELECTED PROJECTS
Worxstance — AI-Powered Career Workspace: an AI career workspace covering resume tailoring, cover letters, skill-gap analysis, mock interviews, and networking outreach, all built against a single Master Profile. Job discovery (multi-source search with AI match scoring) is fully built but currently hidden from users pending job-board legal clearance; offer negotiation is still on the roadmap and not yet built. Every AI feature runs through Gemini 2.5 Flash with defined response schemas. Firestore-based AI usage limits and Firebase per-user data isolation; no billing/payment processing yet (Stripe was planned but never integrated). Stack: React 19, TypeScript, Vite, Firebase, Gemini.

SpecAnalyzer — Construction Drawing & Specification Takeoff: a document-AI pipeline that ingests architectural drawing sets, classifies pages, extracts window/door schedules, and computes glazing quantities deterministically, keeping the language model out of the math. Stack: Next.js, Python FastAPI, PyMuPDF, camelot, AWS Textract, Anthropic Claude.

Tikas Family Digital Home — Self-Hosted AI Platform: a nine-phase, spec-first family platform where private-data questions run entirely on local Ollama models and general questions route to Claude behind an explicit confirm step. Semantic search, role-based permissions, AES-256-GCM encryption, 433 automated tests. Stack: Next.js 15, TypeScript, SQLite, Ollama, Anthropic SDK, Expo.

YouTube Product Crew — Multi-Agent Content Pipeline: parallel research agents across OpenAI, DeepSeek, and Gemini with a synthesis manager agent, producing scripts, SEO keywords, and titles from a single product input. Stack: Python, CrewAI, OpenAI, DeepSeek, Gemini, DALL-E 3, Gradio.

AI News Aggregator — Blog Post Generator: a CrewAI pipeline where a researcher agent and a blog-writer agent turn a single topic into a 600-1000 word, conversational blog post formatted with Jekyll front matter for direct GitHub Pages publishing. Runs via a web interface or the command line; costs about $0.01-0.05 per post on GPT-4o-mini. Stack: Python, CrewAI, GPT-4o-mini, FastAPI.

PROFESSIONAL EXPERIENCE
Technical Support Analyst — IT Support Center (Remote), Aug 2021–Oct 2024 and Mar 2025–Present. Resolves escalated technical issues, serves as Senior Technical Advisor and trainer, handles 15–35 support contacts per day.
Technical Support Analyst — Availity (contract via Insight Global), Oct 2024–Feb 2025. Supported healthcare providers and payers on a national claims platform.
Customer Support Technician — Bank of America (contract via Diversant, LLC), 2017–2018. Delivered end-user support to a 20,000+ employee population.
Customer Service Representative — Lee & Cates Glass, Inc., 2007–2017 and 2018–2019. Primary point of contact for accounts representing over 50% of company sales; managed multi-million-dollar commercial glazing projects.

EDUCATION & CERTIFICATIONS
Bachelor of Science, Information Technology — University of Phoenix.
Certificate in Advanced Business Analytics — University of Phoenix.
United States Navy — Nuclear Machinist's Mate.
Certifications: Microsoft Azure Fundamentals (AZ-900), Google Cybersecurity Certificate, Advanced Business Analytics, Foundations of Project Management, Agile and Scrum Development, CompTIA Security+ (in progress).`;

module.exports = { BIO_TEXT, RESUME_TEXT };
