import spacy
from collections import Counter

nlp = spacy.load("en_core_web_md")

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

 # Alias mapping for common tech terms
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
    doc = nlp(text.lower())
    keyword_candidates = []

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()

        if len(phrase.split()) > 3 or len(phrase) < 3:
            continue

        if any(
            word.lemma_ in GENERIC_NOUNS or word.text in BAD_KEYWORDS
            for word in chunk
            if word.pos_ == "NOUN"
        ):
            continue

        keyword_candidates.append(phrase)

    term_freq = Counter(keyword_candidates)
    return [term for term, _ in term_freq.most_common(top_n)]
