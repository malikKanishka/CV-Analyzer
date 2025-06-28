from flask import Blueprint, request, render_template, redirect, flash
from .utils.keyword_extraction import extract_keywords
from .utils.resume_matcher import find_keyword_presence
from .utils.visualizer import plot_pie_chart
import os

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
@main.route('/ats.html', methods=['GET', 'POST'])
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
            filepath = os.path.join('uploads', file.filename)
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
