import spacy
import pdfplumber
from flask import flash
import json

nlp = spacy.load("en_core_web_md")

# Load keyword config from JSON
with open("app/utils/keyword_config.json", "r") as f:
    config = json.load(f)

GENERIC_NOUNS = set(config["GENERIC_NOUNS"])
BAD_KEYWORDS = set(config["BAD_KEYWORDS"])
ALIAS_MAP = {key: set(values) for key, values in config["ALIAS_MAP"].items()}

def is_valid_keyword(kw):
    if kw not in GENERIC_NOUNS and kw not in BAD_KEYWORDS:
        return True
    else:
        return False

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
