import spacy
from collections import Counter
import json

nlp = spacy.load("en_core_web_md")

# Load keyword config from JSON
with open("app/utils/keyword_config.json", "r") as f:
    config = json.load(f)

GENERIC_NOUNS = set(config["GENERIC_NOUNS"])
BAD_KEYWORDS = set(config["BAD_KEYWORDS"])
ALIAS_MAP = {key: set(values) for key, values in config["ALIAS_MAP"].items()}

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
