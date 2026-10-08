# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility and gives concrete improvement suggestions using Google's Gemini Flash model.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for keyword matching
- ATS score out of 100 with a breakdown (keywords, formatting, impact, skills/education, contact info, readability)
- Strengths, weaknesses, missing keywords
- Prioritised improvements and before/after rewrite examples
- Download the report as JSON

## Tech stack
- [Streamlit](https://streamlit.io) for the UI
- [google-genai](https://pypi.org/project/google-genai/) SDK with a Gemini Flash model
- `pypdf` and `python-docx` for text extraction

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt
```

Get a free API key from https://aistudio.google.com/apikey, then provide it in ONE of these ways:

1. Environment variable: `export GEMINI_API_KEY="your_key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your_key"`)
2. File `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
3. Type it into the sidebar when the app runs.

```bash
streamlit run app.py
```

## Configuration
| Name | Purpose | Default |
|------|---------|---------|
| `GEMINI_API_KEY` | Your Gemini API key | none (required) |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.8-flash` |

If the model is unavailable, the app automatically falls back to `gemini-flash-latest`. If Google retires a model name again, set `GEMINI_MODEL` (env var or Streamlit secret) to a current Flash model from https://ai.google.dev/gemini-api/docs/models.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
5. Click **Deploy**.

## Notes and limitations
- Scanned/image-only PDFs have no extractable text. ATS software cannot read them either, so the app asks for a text-based file.
- The score is an AI estimate, not the output of a real ATS. Use it as guidance.
- Resumes are sent to the Gemini API for analysis. Do not upload data you are not comfortable sharing.

## Project structure
```
ats-resume-checker/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
