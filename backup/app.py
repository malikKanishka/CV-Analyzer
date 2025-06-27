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
nlp = spacy.load("en_core_web_sm")

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def extract_keywords(text, top_n=10):
    """Extract top N nouns/proper nouns from job description."""
    doc = nlp(text.lower())
    terms = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and len(token.text) > 2]
    return [term for term, _ in Counter(terms).most_common(top_n)]

def find_keyword_presence(pdf_path, keywords):
    """Check which of the target keywords are present in the PDF."""
    matched = set()
    unmatched = set(keywords)

    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + " "

        doc = nlp(full_text.lower())
        lemmatized_tokens = set(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

        for keyword in keywords:
            if keyword.lower() in lemmatized_tokens:
                matched.add(keyword)
                unmatched.discard(keyword)

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
