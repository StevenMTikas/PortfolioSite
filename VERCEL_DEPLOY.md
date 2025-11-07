# Vercel Deployment Guide

This portfolio site is a **static HTML site** and should be deployed as such on Vercel.

## Quick Fix for "No fastapi entrypoint found" Error

Vercel is detecting Python files and trying to deploy as a FastAPI app. Here's how to fix it:

### Option 1: Configure in Vercel Dashboard (Recommended)

1. Go to your Vercel project dashboard
2. Click on **Settings** → **General**
3. Under **Framework Preset**, select **"Other"** or **"Static Site"**
4. Set **Build Command** to: (leave empty)
5. Set **Output Directory** to: `.` (or leave empty)
6. Set **Install Command** to: (leave empty)
7. Save and redeploy

### Option 2: Use vercel.json (Already Created)

The `vercel.json` file has been created to configure this as a static site. If Vercel still detects Python:

1. Make sure `vercel.json` is committed to your repository
2. In Vercel dashboard, go to **Settings** → **General**
3. Manually override **Framework Preset** to **"Other"**
4. Redeploy

## Files Created

- `vercel.json` - Configuration file telling Vercel this is a static site
- `.vercelignore` - Excludes Python files from deployment (though Vercel may still scan them)

## What Gets Deployed

Only these files/folders are needed for the static site:
- `index.html` (main file)
- `styles.css`
- `images/` folder
- `assets/` folder (if needed for static assets)

Python files (`chatbot_server.py`, `requirements.txt`, etc.) are for the Render deployment only and are not needed for Vercel.

## After Deployment

1. Test the site loads correctly
2. Test the chatbot bubble connects to `https://portfoliosite-6lw5.onrender.com/api/ask`
3. Test all external links work
4. Update CORS on Render to allow your Vercel domain (if needed)

