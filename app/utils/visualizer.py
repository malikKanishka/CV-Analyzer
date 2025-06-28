import matplotlib.pyplot as plt
import matplotlib
import io
import base64

matplotlib.use('Agg')
plt.ioff()

def plot_pie_chart(matched_count, total_keywords):
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
