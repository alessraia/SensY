import nltk
import numpy as np

# Importa i metodi comuni
from common_functions import (
    shared_count_pos_tags,
    shared_count_sensitive_words,
    shared_generate_single_embedding,
    sia
)

# List of sensitive words
SENSITIVE_WORDS = [
    "war", "violence", "terrorism", "racism", "sexism", "discrimination",
    "abortion", "religion", "politics", "LGBTQ", "poverty", "inequality",
    "slavery", "abuse", "murder", "assault", "genocide", "immigration",
    "gun", "shooting", "protest", "riot", "extremism", "corruption",
    "feminism", "oppression", "hate", "harassment",
    "torture", "massacre", "bombing", "hostage", "kidnapping", "execution",
    "lynching", "cruelty", "bloodshed", "atrocity", "militia", "paramilitary",
    "landmine", "nuclear", "bioweapon", "chemical_weapon", "airstrike", "firing_squad",
    "homophobia", "transphobia", "antisemitism", "islamophobia", "xenophobia",
    "ageism", "ableism", "bigotry", "misogyny", "misandry", "ethnic_cleansing",
    "white_supremacy", "neo_nazi", "kkk", "segregation",
    "apartheid", "junta", "coup", "fascism", "authoritarianism", "dictatorship",
    "martial_law", "censorship", "propaganda", "repression", "surveillance",
    "blacklist", "genocidal",
    "human_trafficking", "forced_marriage", "child_labor", "child_soldier",
    "female_genital_mutilation", "honor_killing", "dowry_death", "bride_burning",
    "acid_attack", "domestic_violence", "pedophilia", "sexual_exploitation",
    "rape", "incest", "molestation", "stalking",
    "self_harm", "suicide", "depression", "anorexia", "bulimia", "overdose",
    "drug_cartel", "drug_lord", "opioid", "heroin", "cocaine", "methamphetamine",
    "mafia", "organized_crime", "cartel", "gang_violence", "money_laundering",
    "racketeering", "human_smuggling",
    "blasphemy", "heresy", "sectarian", "jihad", "fatwa", "religious_persecution",
    "totalitarianism", "apartheid", "ethnostate", "gerrymandering", "coup_d_etat",
    "political_prisoner", "dissident", "black_op", "state_sponsored_terror",
    "hate_speech", "racial_slur", "nazi_symbol", "holocaust_denial", "ethnic_slur",
    "forced_sterilization", "mass_grave", "ethnic_tension", "hate_crime",
    "radicalization", "extremist_cell", "isis", "al_qaeda", "white_nationalism",
    "sex_trafficking", "child_pornography", "sexual_slavery", "coercion", "grooming",
    "sweatshop", "bonded_labor", "indentured_servitude",
    "systemic_racism", "institutional_discrimination", "social_exclusion",
    "marginalization", "caste_system",
    "human_rights_abuse", "war_crime", "crime_against_humanity", "re-education_camp",
    "concentration_camp", "child_abuse", "forced_displacement", "refugee_crisis"
]


def count_pos_tags(text, pos_tags):
    return shared_count_pos_tags(text, pos_tags)

def count_sensitive_words(text):
    return shared_count_sensitive_words(text, SENSITIVE_WORDS)

def generate_bert_embedding_single(text):
    return shared_generate_single_embedding(text)

def extract_features_single(question_en):
    """
    Extract features from a single question (syntactic + BERT).
    """
    # Ensure question_en is a string
    question_en = str(question_en)

    num_unique_words = len(set(nltk.word_tokenize(question_en)))
    num_verbs = count_pos_tags(question_en, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
    num_adjectives = count_pos_tags(question_en, ["JJ", "JJR", "JJS"])
    num_nouns = count_pos_tags(question_en, ["NN", "NNS", "NNP", "NNPS"])
    num_sensitive = count_sensitive_words(question_en)

    # Sentiment
    sentiment = sia.polarity_scores(question_en)["compound"]

    syntactic_semantic = np.array([
        num_unique_words,
        num_verbs,
        num_adjectives,
        num_nouns,
        num_sensitive,
        sentiment
    ]).reshape(1, -1)

    # BERT embedding (768 dimensioni)
    bert_emb = generate_bert_embedding_single(question_en)  # (1, 768)

    features = np.hstack([syntactic_semantic, bert_emb])
    return features