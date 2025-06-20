"""
ATS Score Analyzer - Flask Application
=======================================
This module handles ATS (Applicant Tracking System) functionality.
It processes resumes against job descriptions to evaluate keyword matches.

Features:
---------
- Extracts keywords from job descriptions using spaCy NLP.
- Parses PDF resumes and checks keyword presence.
- Calculates match percentage and generates a pie chart.
- Returns insights (matched and unmatched keywords).
- Flask-integrated with full error handling.

Author: Kanishka Malik
License: MIT
"""

from flask import Flask, request, render_template, redirect, url_for, flash
import os
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import PyPDF2
import spacy
import warnings

# Disable GUI backend for headless chart rendering
matplotlib.use('Agg')
plt.ioff()
warnings.filterwarnings("ignore")

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def extract_technical_terms(job_description, num_keywords=10):
    """Extracts most common nouns/proper nouns from the job description."""
    doc = nlp(job_description.lower())
    terms = [token.text for token in doc if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop and len(token.text) > 2]
    counts = Counter(terms)
    return [keyword for keyword, _ in counts.most_common(num_keywords)]

def search_words_in_pdf(pdf_path, search_terms):
    """Searches for keyword presence in resume PDF using spaCy NLP."""
    matched_terms = set()
    unmatched_terms = set(search_terms)

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            resume_doc = nlp(text.lower())
            resume_tokens = set([token.lemma_ for token in resume_doc if not token.is_stop and token.is_alpha])

            for term in search_terms:
                term_doc = nlp(term.lower())
                for token in term_doc:
                    if token.lemma_ in resume_tokens:
                        matched_terms.add(term)
                        unmatched_terms.discard(term)
                        break
    except Exception as e:
        flash(f"Error reading PDF: {e}", 'danger')
        return set(), set()

    return matched_terms, unmatched_terms


def plot_pie_chart(matched_percentage):
    """Generates pie chart for matched vs unmatched percentage."""
    matched = max(0, min(100, round(matched_percentage)))
    unmatched = 100 - matched

    if matched == 0 and unmatched == 0:
        unmatched = 1  # Prevent crash

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
    return chart_url

@app.route('/', methods=['GET', 'POST'])
@app.route('/ats.html', methods=['GET', 'POST'])
def ats():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        file = request.files.get('resume')

        if not job_description:
            flash("Please enter a job description.", 'warning')
            return redirect(request.url)

        if not file or file.filename == '':
            flash("Please upload a valid resume file.", 'warning')
            return redirect(request.url)

        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            technical_terms = extract_technical_terms(job_description)
            matched_terms, unmatched_terms = search_words_in_pdf(file_path, technical_terms)
            matched_count = len(matched_terms)
            total_terms = len(technical_terms)
            match_percentage = (matched_count / total_terms) * 100 if total_terms > 0 else 0
            chart_url = plot_pie_chart(match_percentage)

            flash(f'Your resume matches {match_percentage:.2f}% of key skills.', 'success')
            flash("Matched Keywords: " + ", ".join(sorted(matched_terms)), 'info')

            return render_template('ats.html',
                                   chart_url=chart_url,
                                   matched_terms=matched_terms,
                                   matched_count=matched_count,
                                   match_percentage=match_percentage,
                                   technical_terms=technical_terms,
                                   unmatched_terms=unmatched_terms)

        except Exception as e:
            flash(f"Something went wrong: {e}", 'danger')
            return redirect(request.url)

    return render_template('ats.html')

if __name__ == '__main__':
    app.run(debug=True)
