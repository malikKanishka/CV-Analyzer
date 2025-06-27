from flask import Flask, request, render_template, redirect, flash, url_for
import os
import spacy
import pdfplumber
from collections import Counter
import warnings
import io
import base64
import matplotlib.pyplot as plt
import matplotlib

# Set up environment
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
plt.ioff()

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
nlp = spacy.load("en_core_web_md")

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
GENERIC_NOUNS = {
    # Resume-related nouns
    "candidate", "employee", "applicant", "individual", "person", "people",

    # Organizational nouns
    "company", "organization", "department", "team", "group", "firm", "business", "office",

    # Common abstract nouns
    "experience", "ability", "knowledge", "skill", "competence", "understanding", "background",

    # Job-related nouns
    "role", "position", "responsibility", "requirement", "qualification", "title", "career", "job", "opportunity", "function", "vacancy", "opening",

    # Work environment nouns
    "environment", "culture", "workplace", "setting", "location", "field", "area", "domain",

    # Task/process nouns
    "task", "process", "activity", "operation", "workflow", "procedure", "step", "effort",

    # Project nouns
    "project", "initiative", "assignment", "program", "plan", "goal", "objective",

    # Communication nouns
    "communication", "discussion", "conversation", "presentation", "meeting",

    # Time nouns
    "time", "schedule", "deadline", "period", "duration", "shift",

    # Generic tool/framework nouns
    "tool", "platform", "framework", "system", "software", "hardware", "technology",

    # Other general-purpose nouns
    "value", "success", "growth", "support", "service", "solution", "result", "trend", "focus", "opinion", "insight", "resource"
}

    # Blacklist of generic or meaningless keywords to skip
BAD_KEYWORDS = {
        "you", "your", "that", "those", "familiarity", "proficiency", "completion",
        "knowledge", "experience", "ability", "certifications", "medium", "basic", "ideas",
        "self", "relevant", "tata", "life", "dynamic", "driven", "motivated", "fast-paced"
    }

 # 🔁 Alias mapping for common tech terms
ALIAS_MAP = {
        "react.js": {"reactjs", "react js"},
        "reactjs": {"react.js", "react js"},
        "javascript": {"js"},
        "typescript": {"ts"},
        "nlp": {"natural language processing"},
        "natural language processing": {"nlp"},
        "git/github": {"git", "github"},
        "machine learning": {"ML"},
        "deep learning": {"DL"},
        "DL": {"deep learning"},
        "ML": {"machine learning"}
    }

def extract_keywords(text, top_n=10):
    """Extract top N clean and relevant keyword phrases from job description."""
    doc = nlp(text.lower())
    keyword_candidates = []

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()

        # Skip phrases that are too long or too short
        if len(phrase.split()) > 3 or len(phrase) < 3:
            continue

        # Skip phrases containing bad/generic words
        if any(
            word.lemma_ in GENERIC_NOUNS or word.text in BAD_KEYWORDS
            for word in chunk
            if word.pos_ == "NOUN"
        ):
            continue

        keyword_candidates.append(phrase)

    # Deduplicate & count frequency
    term_freq = Counter(keyword_candidates)
    return [term for term, _ in term_freq.most_common(top_n)]

# Remove bad or irrelevant keywords before matching
def is_valid_keyword(kw):
    if kw not in GENERIC_NOUNS and kw not in BAD_KEYWORDS:
        return True 

def find_keyword_presence(pdf_path, keywords):
    """Check which target keywords are present in the PDF using partial, alias, and semantic matching."""

    keywords = [kw for kw in keywords if is_valid_keyword(kw)]
    matched = set()
    unmatched = set(keywords)

    try:
        # 📄 Extract full resume text from PDF
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + " "

        # 🧠 Process with spaCy
        doc = nlp(full_text.lower())

        # Create clean set of lemmatized tokens from resume
        lemmatized_tokens = set(
            token.lemma_.lower() for token in doc
            if not token.is_stop and not token.is_punct and len(token.text.strip()) > 2
        )

        # 🔍 For each keyword in JD
        for keyword in keywords:
            keyword_clean = keyword.strip().lower()

            # 🛑 Skip unwanted keywords
            if (
                keyword_clean in BAD_KEYWORDS
                or len(keyword_clean) < 3
                or len(keyword_clean.split()) > 4
                or all(word in BAD_KEYWORDS for word in keyword_clean.split())
            ):
                continue

            # 🔁 Generate alias variations for this keyword
            aliases = {keyword_clean}
            for base, variants in ALIAS_MAP.items():
                if keyword_clean == base or keyword_clean in variants:
                    aliases.update(variants)
                    aliases.add(base)

            # 🔄 Try to match with any alias form
            found_match = False
            for token_text in lemmatized_tokens:
                # Check partial match
                if any(alias in token_text or token_text in alias for alias in aliases):
                    matched.add(keyword)
                    unmatched.discard(keyword)
                    found_match = True
                    break

                # Check semantic similarity
                for alias in aliases:
                    alias_doc = nlp(alias)
                    token_doc = nlp(token_text)
                    if alias_doc.vector_norm and token_doc.vector_norm:
                        if alias_doc.similarity(token_doc) >= 0.90:
                            matched.add(keyword)
                            unmatched.discard(keyword)
                            found_match = True
                            break
                if found_match:
                    break

    except Exception as e:
        flash(f"Error reading PDF: {e}", 'danger')
        return set(), set()

    return matched, unmatched


def plot_pie_chart(matched_count, total_keywords):
    """Return base64-encoded pie chart image."""
    matched_percentage = (matched_count / total_keywords) * 100 if total_keywords else 0
    matched = round(matched_percentage)
    unmatched = 100 - matched

    if matched == 0 and unmatched == 0:
        unmatched = 1

    plt.figure(figsize=(6, 6))
    plt.pie(
        [matched, unmatched],
        labels=['Matched Terms', 'Missing Terms'],
        colors=['#4CAF50', '#FF6B6B'],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title('Resume Match Analysis')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return chart_url, matched_percentage

@app.route('/', methods=['GET', 'POST'])
@app.route('/ats.html', methods=['GET', 'POST'])
def ats():
    if request.method == 'POST':
        job_description = request.form.get("job_description")
        file = request.files.get("resume")

        if not job_description:
            flash("Please enter a job description.", "warning")
            return redirect(request.url)

        if not file or file.filename == '':
            flash("Please upload a valid resume file.", "warning")
            return redirect(request.url)

        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            keywords = extract_keywords(job_description)
            matched_terms, unmatched_terms = find_keyword_presence(filepath, keywords)
            matched_count = len(matched_terms)
            total_terms = len(keywords)
            chart_url, match_percentage = plot_pie_chart(matched_count, total_terms)

            flash(f'Your resume matches {match_percentage:.2f}% of key skills.', 'success')

            return render_template(
                'ats.html',
                keywords=keywords,
                matched_terms=sorted(matched_terms),
                unmatched_terms=sorted(unmatched_terms),
                matched_count=matched_count,
                total_terms=total_terms,
                match_percentage=match_percentage,
                chart_url=chart_url
            )

        except Exception as e:
            flash(f"Something went wrong: {e}", "danger")
            return redirect(request.url)

    return render_template("ats.html")

if __name__ == '__main__':
    app.run(debug=True)
