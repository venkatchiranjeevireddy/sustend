import os
import re
import time
import csv
import requests
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.exceptions import HTTPException

try:
    # Optional: load environment variables from a local .env file if present
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, ignore; env vars may already be set
    pass

# ========================
# CONFIGURATION
# ========================
# Set your Groq API key here (or use: export GROQ_API_KEY="your_key" in terminal)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq API endpoint
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "call_analysis.csv")

# Flask app
app = Flask(__name__)

# Limits and simple rate limiting
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "4000"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))  # requests
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "300"))  # seconds
_RATE_BUCKET = {}

# ========================
# HELPER: Call Groq API
# ========================
def call_groq(prompt):
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in your environment or .env file.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    # Retry with exponential backoff
    last_err = None
    for attempt in range(4):
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as req_err:
            last_err = req_err
            sleep_s = (2 ** attempt) * 0.5
            time.sleep(sleep_s)
    raise RuntimeError(f"Groq API request failed after retries: {last_err}")


def redact_pii(text: str) -> str:
    if not text:
        return text
    # Email
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    # Phone numbers (simple patterns)
    text = re.sub(r"\b\+?\d[\d\s().-]{7,}\b", "[REDACTED_PHONE]", text)
    # Credit card (very rough)
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD]", text)
    return text


def within_size_limit(text: str) -> str:
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    return text[:MAX_TRANSCRIPT_CHARS]


def rate_limit_or_raise(key: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    bucket = _RATE_BUCKET.setdefault(key, [])
    # drop old
    bucket[:] = [ts for ts in bucket if ts >= window_start]
    if len(bucket) >= RATE_LIMIT_MAX:
        raise RuntimeError("Rate limit exceeded. Please try again later.")
    bucket.append(now)

# ========================
# ROUTE: Home page
# ========================
@app.route("/", methods=["GET", "POST"])
def home():
    error_message = None
    result = None
    if request.method == "POST":
        client_key = request.remote_addr or "anonymous"
        try:
            rate_limit_or_raise(client_key)
        except Exception as ex:
            error_message = str(ex)
            return render_template_string(
                TEMPLATE,
                transcript=None,
                summary=None,
                sentiment=None,
                timestamp=None,
                error=error_message
            )

        transcript = (request.form.get("transcript") or "").strip()
        try:
            result = analyze_transcript(transcript)
        except Exception as ex:
            error_message = str(ex)

    return render_template_string(
        TEMPLATE,
        transcript=(result["transcript"] if result else None),
        summary=(result["summary"] if result else None),
        sentiment=(result["sentiment"] if result else None),
        timestamp=(result["timestamp"] if result else None),
        error=error_message
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    try:
        client_key = request.remote_addr or "anonymous"
        rate_limit_or_raise(client_key)
        result = analyze_transcript(transcript)
        return jsonify({"ok": True, **result})
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 400


@app.route("/history", methods=["GET"])
def history():
    try:
        if not os.path.isfile(CSV_PATH):
            table_html = "<em>No entries yet.</em>"
        else:
            df = pd.read_csv(CSV_PATH)
            # Ensure expected columns exist for display
            expected_cols = ["Transcript", "Summary", "Sentiment", "Timestamp"]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = ""
            table_html = df[expected_cols].to_html(classes="table", index=False, escape=False)
    except Exception as ex:
        table_html = f"<pre style='color:red;'>Failed to read history: {ex}</pre>"

    return render_template_string(HISTORY_TEMPLATE, table_html=table_html)


@app.route("/download", methods=["GET"])
def download_csv():
    if not os.path.isfile(CSV_PATH):
        return jsonify({"ok": False, "error": "No CSV available yet."}), 404
    return send_file(CSV_PATH, as_attachment=True, download_name="call_analysis.csv")

# ========================
# SIMPLE HTML FORM (UI)
# ========================
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Call Transcript Analyzer</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <style>
    .badge-positive { background: #e6ffed; color: #076e2e; }
    .badge-neutral  { background: #f0f4ff; color: #273e74; }
    .badge-negative { background: #ffecec; color: #8a0000; }
    .char-count { font-size: 12px; color: #6c757d; }
    .copy-btn { white-space: nowrap; }
  </style>
  <link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css" />
</head>
<body>
  <nav class="navbar navbar-expand-lg bg-light border-bottom">
    <div class="container">
      <a class="navbar-brand" href="/">📞 Call Analyzer</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#nav" aria-controls="nav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="nav">
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
          <li class="nav-item"><a class="nav-link" href="/history">History</a></li>
          <li class="nav-item"><a class="nav-link" href="/download">Download CSV</a></li>
        </ul>
      </div>
    </div>
  </nav>

  <main class="container my-4">
    {% if error %}
      <div class="alert alert-danger" role="alert">{{ error }}</div>
    {% endif %}

    <div class="row g-4">
      <div class="col-12 col-lg-6">
        <div class="card shadow-sm">
          <div class="card-body">
            <h5 class="card-title mb-3">Analyze a Transcript</h5>
            <form method="POST" onsubmit="return onSubmitAnalyze(this)">
              <div class="mb-2 d-flex justify-content-between">
                <label for="transcript" class="form-label mb-0">Paste Transcript</label>
                <span class="char-count" id="charCount">0 chars</span>
              </div>
              <textarea class="form-control" name="transcript" id="transcript" rows="8" required oninput="updateCount()"></textarea>
              <div class="d-flex gap-2 mt-3">
                <button id="analyzeBtn" type="submit" class="btn btn-primary">
                  <span class="spinner-border spinner-border-sm me-2 d-none" id="btnSpinner" role="status" aria-hidden="true"></span>
                  Analyze
                </button>
                <button type="button" class="btn btn-outline-secondary" onclick="fillExample()">Use Example</button>
                <button type="button" class="btn btn-outline-dark" onclick="toggleDarkMode()">Dark Mode</button>
              </div>
              <div class="form-text">Provide a short customer ↔ agent dialogue. Sensitive data will be sent to the API.</div>
            </form>
          </div>
        </div>
      </div>

      <div class="col-12 col-lg-6">
        {% if transcript %}
        <div class="card shadow-sm">
          <div class="card-body">
            <h5 class="card-title">Results</h5>
            <div class="mb-3">
              <label class="form-label">Transcript</label>
              <div class="input-group">
                <textarea class="form-control" rows="4" readonly id="outTranscript">{{ transcript }}</textarea>
                <button class="btn btn-outline-secondary copy-btn" type="button" onclick="copyText('outTranscript')">Copy</button>
              </div>
            </div>
            <div class="mb-3">
              <label class="form-label">Summary</label>
              <div class="input-group">
                <textarea class="form-control" rows="3" readonly id="outSummary">{{ summary }}</textarea>
                <button class="btn btn-outline-secondary copy-btn" type="button" onclick="copyText('outSummary')">Copy</button>
              </div>
            </div>
            <div class="mb-2">
              <label class="form-label">Sentiment</label>
              <div>
                {% set s = (sentiment or '').lower() %}
                {% if 'pos' in s %}<span class="badge badge-positive px-2 py-1">{{ sentiment }}</span>
                {% elif 'neg' in s %}<span class="badge badge-negative px-2 py-1">{{ sentiment }}</span>
                {% else %}<span class="badge badge-neutral px-2 py-1">{{ sentiment }}</span>{% endif %}
              </div>
            </div>
            {% if timestamp %}<div class="text-muted">Timestamp: {{ timestamp }}</div>{% endif %}
          </div>
        </div>
        {% else %}
        <div class="alert alert-info">Paste a transcript and click Analyze to see results here.</div>
        {% endif %}
      </div>
    </div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    function fillExample() {
      const ex = `Customer: Hi, I was trying to book a slot yesterday but the payment failed.\nAgent: Sorry to hear that. Could you please share your booking ID?\nCustomer: It's 12345.\nAgent: Thanks, I can see a declined transaction. I'll re-trigger the payment link.`;
      const ta = document.getElementById('transcript');
      ta.value = ex;
      updateCount();
      ta.focus();
    }

    function copyText(id) {
      const el = document.getElementById(id);
      el.select();
      el.setSelectionRange(0, 99999);
      navigator.clipboard.writeText(el.value);
    }

    function updateCount() {
      const ta = document.getElementById('transcript');
      const cc = document.getElementById('charCount');
      cc.textContent = (ta.value || '').length + ' chars';
    }

    function onSubmitAnalyze(form) {
      const btn = document.getElementById('analyzeBtn');
      const spinner = document.getElementById('btnSpinner');
      btn.disabled = true;
      spinner.classList.remove('d-none');
      return true;
    }

    function toggleDarkMode() {
      document.body.classList.toggle('bg-dark');
      document.body.classList.toggle('text-white');
    }
  </script>
</body>
</html>
"""


HISTORY_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Call Transcript History</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
  <nav class="navbar navbar-expand-lg bg-light border-bottom">
    <div class="container">
      <a class="navbar-brand" href="/">📞 Call Analyzer</a>
      <div class="collapse navbar-collapse" id="nav">
        <ul class="navbar-nav ms-auto">
          <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
          <li class="nav-item"><a class="nav-link" href="/history">History</a></li>
          <li class="nav-item"><a class="nav-link" href="/download">Download CSV</a></li>
        </ul>
      </div>
    </div>
  </nav>
  <main class="container my-4">
    <h2 class="mb-3">📜 Analysis History</h2>
    <div class="table-responsive">
      {{ table_html|safe }}
    </div>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


# ========================
# CORE ANALYSIS LOGIC
# ========================
def analyze_transcript(transcript: str) -> dict:
    if not transcript:
        raise ValueError("Transcript cannot be empty.")

    # Enforce size limit
    original_length = len(transcript)
    transcript = within_size_limit(transcript)
    was_truncated = original_length != len(transcript)

    # Redact PII before sending to Groq
    redacted_for_model = redact_pii(transcript)

    # Prompt Groq for summary
    summary_prompt = (
        "Summarize this customer support call in 2–3 concise sentences. "
        "Focus on the customer's problem and any resolution steps.\n\n"
        f"Transcript:\n{transcript}"
    )
    summary = call_groq(summary_prompt.replace(f"{transcript}", redacted_for_model))

    # Prompt Groq for sentiment
    sentiment_prompt = (
        "Classify the customer's sentiment in ONE WORD from this set: "
        "Positive, Neutral, Negative. Only output the single word.\n\n"
        f"Transcript:\n{transcript}"
    )
    sentiment = call_groq(sentiment_prompt.replace(f"{transcript}", redacted_for_model))

    # Print results to console for the assignment requirement
    print("\n=== Call Analysis ===")
    print("Transcript:", transcript)
    print("Summary:", summary)
    print("Sentiment:", sentiment)

    # Save into CSV (append, create header if missing) with timestamp
    file_exists = os.path.isfile(CSV_PATH)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Transcript", "Summary", "Sentiment", "Timestamp"])
        writer.writerow([transcript, summary, sentiment, timestamp])

    return {
        "transcript": transcript + ("\n[TRUNCATED]" if was_truncated else ""),
        "summary": summary,
        "sentiment": sentiment,
        "timestamp": timestamp,
    }


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Call Transcript Analyzer API",
            "version": "1.0.0"
        },
        "paths": {
            "/api/analyze": {
                "post": {
                    "summary": "Analyze transcript",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "transcript": {"type": "string"}
                                    },
                                    "required": ["transcript"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "ok": {"type": "boolean"},
                                            "transcript": {"type": "string"},
                                            "summary": {"type": "string"},
                                            "sentiment": {"type": "string"},
                                            "timestamp": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad Request"
                        }
                    }
                }
            }
        }
    }
    return jsonify(spec)


@app.route("/docs", methods=["GET"])
def docs():
    return render_template_string("""
    <!doctype html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>API Docs</title>
      <link rel=\"stylesheet\" href=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui.css\" />
    </head>
    <body>
      <div id=\"swagger-ui\"></div>
      <script src=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js\"></script>
      <script>
        window.onload = () => {
          window.ui = SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui'
          });
        };
      </script>
    </body>
    </html>
    """)

# ========================
# MAIN
# ========================
@app.errorhandler(Exception)
def handle_exceptions(error):
    if isinstance(error, HTTPException):
        return jsonify({"ok": False, "error": error.description}), error.code
    return jsonify({"ok": False, "error": str(error)}), 500


if __name__ == "__main__":
    # Allow overriding the port via env var
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, port=port)
