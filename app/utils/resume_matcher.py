import spacy
import pdfplumber
from flask import flash

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

def is_valid_keyword(kw):
    return kw not in GENERIC_NOUNS and kw not in BAD_KEYWORDS

def find_keyword_presence(pdf_path, keywords):
    keywords = [kw for kw in keywords if is_valid_keyword(kw)]
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
        lemmatized_tokens = set(
            token.lemma_.lower() for token in doc
            if not token.is_stop and not token.is_punct and len(token.text.strip()) > 2
        )

        for keyword in keywords:
            keyword_clean = keyword.strip().lower()

            if (
                keyword_clean in BAD_KEYWORDS
                or len(keyword_clean) < 3
                or len(keyword_clean.split()) > 4
                or all(word in BAD_KEYWORDS for word in keyword_clean.split())
            ):
                continue

            aliases = {keyword_clean}
            for base, variants in ALIAS_MAP.items():
                if keyword_clean == base or keyword_clean in variants:
                    aliases.update(variants)
                    aliases.add(base)

            found_match = False
            for token_text in lemmatized_tokens:
                if any(alias in token_text or token_text in alias for alias in aliases):
                    matched.add(keyword)
                    unmatched.discard(keyword)
                    break

                for alias in aliases:
                    alias_doc = nlp(alias)
                    token_doc = nlp(token_text)
                    if alias_doc.vector_norm and token_doc.vector_norm:
                        if alias_doc.similarity(token_doc) >= 0.90:
                            matched.add(keyword)
                            unmatched.discard(keyword)
                            break

    except Exception as e:
        flash(f"Error reading PDF: {e}", 'danger')
        return set(), set()

    return matched, unmatched
