"""
ATS Score Analyzer - Flask Application
=======================================
This module handles ATS (Applicant Tracking System) functionality.
It processes resumes against job descriptions to evaluate keyword matches.

Features:
---------
- Extracts keywords from job descriptions using spaCy NLP.
- Parses PDF resumes and finds keyword matches.
- Calculates match percentage and generates a pie chart.
- Returns insights (strong and weak points).
- Fully Flask-integrated with edge case handling.

Author: Kanishka Malik
License: MIT
"""

from flask import Flask, request, render_template, redirect, url_for, flash, session
from functools import wraps
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

appp = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

appp.secret_key = 'your_secret_key'
appp.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(appp.config['UPLOAD_FOLDER']):
    os.makedirs(appp.config['UPLOAD_FOLDER'])

def extract_technical_terms(job_description, num_keywords=10):
    """Extracts the most common nouns/proper nouns from the job description."""
    doc = nlp(job_description.lower())
    terms = [token.text for token in doc if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop and len(token.text) > 2]
    counts = Counter(terms)
    return [keyword for keyword, _ in counts.most_common(num_keywords)]

def search_words_in_pdf(pdf_path, search_terms):
    """Searches for keywords in the resume PDF."""
    matched_terms = Counter()
    unmatched_terms = set(search_terms)

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text() or ""
                words = text.split()
                for word in words:
                    word_lower = word.lower()
                    for term in search_terms:
                        if term.lower() in word_lower:
                            matched_terms[term] += 1
                            unmatched_terms.discard(term)
    except Exception as e:
        flash(f"Error reading PDF: {e}", 'danger')
        return Counter(), set()

    return matched_terms, unmatched_terms

def plot_pie_chart(matched_percentage):
    """Creates a pie chart showing match vs unmatched terms."""
    matched = max(0, round(matched_percentage))
    unmatched = max(0, 100 - matched)
    if matched == 0 and unmatched == 0:
        unmatched = 1  # Prevent matplotlib crash

    plt.figure(figsize=(6, 6))
    plt.pie([matched, unmatched], labels=['Matched Terms', 'Missing Terms'],
            colors=['#4CAF50', '#FF6B6B'], autopct='%1.1f%%', startangle=140)
    plt.title('Resume Match Analysis')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return chart_url

@appp.route('/', methods=['GET', 'POST'])
@appp.route('/ats.html', methods=['GET', 'POST'])
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
            file_path = os.path.join(appp.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            technical_terms = extract_technical_terms(job_description)
            matched_terms, unmatched_terms = search_words_in_pdf(file_path, technical_terms)
            matched_count = sum(matched_terms.values())
            total_terms = len(technical_terms)
            match_percentage = (matched_count / total_terms) * 100 if total_terms > 0 else 0
            chart_url = plot_pie_chart(match_percentage)

            strong_points = {term: count for term, count in matched_terms.items() if count > 1}
            weak_points = list(unmatched_terms)

            flash(f'Your resume matches {match_percentage:.2f}% of key skills.', 'success')
            matched_info = [f"{term}: {count} occurrence(s)" for term, count in matched_terms.items()]
            flash("Matched Terms and Occurrences: " + ", ".join(matched_info), 'info')

            return render_template('ats.html', 
                                   chart_url=chart_url, 
                                   matched_terms=matched_terms, 
                                   matched_count=matched_count, 
                                   match_percentage=match_percentage, 
                                   technical_terms=technical_terms, 
                                   strong_points=strong_points, 
                                   weak_points=weak_points)

        except Exception as e:
            flash(f"Something went wrong: {e}", 'danger')
            return redirect(request.url)

    return render_template('ats.html')

if __name__ == '__main__':
    appp.run(debug=True)
