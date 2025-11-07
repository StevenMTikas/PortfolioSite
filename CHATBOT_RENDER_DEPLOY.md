# Chatbot Render Deployment Guide

This guide will help you deploy the chatbot as a standalone web app on Render.

## Prerequisites

1. A Render account (sign up at https://render.com)
2. Your OpenAI API key
3. GitHub repository with your code (or use Render's direct deployment)

## Files Created/Modified

- `chatbot/index.html` - Standalone HTML page for the chatbot
- `chatbot_server.py` - Updated to serve the HTML page and handle Render deployment
- `render.yaml` - Render configuration file
- `requirements.txt` - Python dependencies (should already exist)

## Deployment Steps

### 1. Prepare Your Repository

Make sure these files are in your repository:
- `chatbot_server.py`
- `chatbot/index.html`
- `requirements.txt`
- `render.yaml` (optional, but recommended)
- `assets/MeBot/` folder with your PDF files and bio.txt

### 2. Create a New Web Service on Render

1. Log in to Render dashboard
2. Click "New +" → "Web Service"
3. Connect your GitHub repository (or use public repo URL)
4. Configure the service:
   - **Name**: `chatbot` (or your preferred name)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or specify if chatbot files are in a subdirectory)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python chatbot_server.py`

### 3. Set Environment Variables

In the Render dashboard, go to your service → Environment tab and add:

- **OPENAI_API_KEY**: Your OpenAI API key (required)
- **PORT**: `10000` (Render sets this automatically, but good to have)
- **GITHUB_TOKEN**: (Optional) GitHub Personal Access Token for higher API rate limits
  - Get one from: https://github.com/settings/tokens
  - Without token: 60 requests/hour (may hit limit with many repos)
  - With token: 5000 requests/hour (recommended)
- **ALLOWED_ORIGINS**: (Optional) Comma-separated list of allowed origins for CORS
  - Example: `https://your-portfolio.vercel.app,https://your-domain.com`
  - Leave empty or don't set to allow all origins (for testing)

### 4. Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Start your application
3. Wait for deployment to complete (usually 2-5 minutes)

### 5. Access Your Chatbot

Once deployed, your chatbot will be available at:
- `https://your-service-name.onrender.com`

The root URL (`/`) will serve the chatbot HTML page, and the API will be available at `/api/ask`.

## File Structure

Your deployed structure should look like:
```
.
├── chatbot_server.py      # Main FastAPI server
├── chatbot/
│   └── index.html         # Standalone HTML page
├── assets/
│   └── MeBot/
│       ├── bio.txt
│       ├── Resume.pdf
│       └── LinkedIn_Profile.pdf
├── requirements.txt       # Python dependencies
└── render.yaml           # Render config (optional)
```

## Testing Locally Before Deployment

1. Make sure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your environment variables:
   ```bash
   export OPENAI_API_KEY=your_key_here
   export PORT=8000
   ```

3. Run the server:
   ```bash
   python chatbot_server.py
   ```

4. Open `http://localhost:8000` in your browser
   - You should see the chatbot interface
   - Try asking a question to test the API

## Troubleshooting

### Build Fails

- Check that `requirements.txt` includes all dependencies
- Verify Python version compatibility (Render uses Python 3.9+ by default)
- Check build logs in Render dashboard

### Service Won't Start

- Verify `chatbot_server.py` is in the root directory (or update start command)
- Check that `assets/MeBot/` folder exists with PDF files
- Review startup logs in Render dashboard

### API Key Errors

- Verify `OPENAI_API_KEY` is set correctly in Render environment variables
- Check that the key is valid and has credits
- Review server logs for specific error messages

### GitHub Loading Issues

- If GitHub repositories don't load, check:
  - GitHub username is correct (currently "StevenMTikas")
  - Network connectivity to GitHub API
  - Rate limit status (check logs for rate limit warnings)
  - Consider setting `GITHUB_TOKEN` for higher rate limits
- First question may take longer as GitHub data loads on-demand
- Subsequent questions will be faster as data is cached

### CORS Issues

- If calling from your portfolio site, add your portfolio URL to `ALLOWED_ORIGINS`
- Format: `https://your-portfolio.vercel.app` (comma-separated for multiple)
- Or leave unset to allow all origins (less secure, but easier for testing)

### Slow Initialization

- First startup can take 2-5 minutes as it loads and processes PDFs
- This is normal - subsequent requests will be faster
- Consider using Render's paid plans for better performance

## Updating Your Portfolio Site

Once deployed, update your portfolio site to point to the Render URL:

In `index.html`, change:
```javascript
// FROM:
const response = await fetch('http://localhost:8000/api/ask', {

// TO:
const response = await fetch('https://your-chatbot-service.onrender.com/api/ask', {
```

## Render Free Tier Limitations

- Services spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds (cold start)
- 750 hours/month free (enough for most personal projects)
- Consider paid plans for production use

## Next Steps

1. Test the deployed chatbot
2. Update your portfolio site to use the Render URL
3. Monitor usage and performance in Render dashboard
4. Consider setting up a custom domain (Render supports this)

## Support

- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com
- Check Render dashboard logs for detailed error messages

