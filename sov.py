import streamlit as st
from openai import OpenAI
from transformers import pipeline
import ast
import matplotlib.pyplot as plt
import numpy as np

# Charger les secrets depuis le fichier secrets.toml
openai_key = st.secrets["openai_key"]

# Configuration de l'API OpenAI
client = OpenAI(api_key=openai_key)

# Charger le modèle de sentiment avec une longueur maximale
sentiment_model = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis", max_length=512, truncation=True)

def obtenir_reponse(question):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        temperature=0.2
    )
    return completion.choices[0].message.content

def generer_questions(marque):
    prompt_questions = f"""Génère 3 questions associées à la marque : {marque} et 2 questions associées à son secteur."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_questions}]
    )
    return completion.choices[0].message.content.split("\n")

def extraire_marques(texte):
    prompt_marques = f"""Identifie les marques dans ce texte et donne-moi la liste sous ce format : ["marque1", "marque2"].
    Texte : {texte}"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_marques}],
        temperature=0
    )
    try:
        return ast.literal_eval(completion.choices[0].message.content)
    except (SyntaxError, ValueError):
        return []

def extraire_elements_semantiques(texte):
    prompt_elements = f"""Identifie les éléments sémantiques importants dans ce texte et donne-moi la liste sous ce format : ["prix", "taille", "qualité"].
    Texte : {texte}"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_elements}],
        temperature=0
    )
    try:
        return ast.literal_eval(completion.choices[0].message.content)
    except (SyntaxError, ValueError):
        return []

def analyser_reponse(reponse, marque):
    try:
        # Analyse de sentiment
        sentiment_result = sentiment_model(reponse)[0]
        sentiment = sentiment_result['label']
    except Exception as e:
        st.error(f"Erreur lors de l'analyse de sentiment : {e}")
        sentiment = "Erreur"

    # Extraction des marques
    marques_mentionnees = extraire_marques(reponse)

    # Extraction des éléments sémantiques
    elements_semantiques = extraire_elements_semantiques(reponse)

    # Analyse simple pour démonstration
    mention_marque = marque.lower() in reponse.lower()

    return {
        "mention_marque": mention_marque,
        "sentiment": sentiment,
        "marques_mentionnees": marques_mentionnees,
        "elements_semantiques": elements_semantiques
    }

def comparer_sentiments(analyses):
    sentiments = {"Very Positive": 0, "Positive": 0, "Neutral": 0, "Negative": 0, "Very Negative": 0}
    for analyse in analyses:
        sentiments[analyse["sentiment"]] += 1
    return sentiments

def synthese_marques(analyses):
    marques_count = {}
    for analyse in analyses:
        for marque_mentionnee in analyse["marques_mentionnees"]:
            if marque_mentionnee in marques_count:
                marques_count[marque_mentionnee] += 1
            else:
                marques_count[marque_mentionnee] = 1

    # Garder les 9 marques les plus mentionnées
    top_marques = sorted(marques_count.items(), key=lambda x: x[1], reverse=True)[:9]
    return {marque: count for marque, count in top_marques}

def synthese_elements_semantiques(analyses):
    elements_count = {}
    for analyse in analyses:
        for element in analyse["elements_semantiques"]:
            if element in elements_count:
                elements_count[element] += 1
            else:
                elements_count[element] = 1

    # Trier les éléments sémantiques par ordre décroissant et garder le top 10
    top_elements = sorted(elements_count.items(), key=lambda x: x[1], reverse=True)[:10]
    return {element: count for element, count in top_elements}

# Interface Streamlit
st.image("SlayLLM.jpg", width=200)
st.title("Analyse des Réponses des LLM")

marque = st.text_input("Entrez la marque à analyser :")
if marque:
    questions = generer_questions(marque)
    st.write("**Questions générées :**")
    for question in questions:
        st.write(f"- {question}")

    # Permettre à l'utilisateur de modifier les questions
    st.write("**Modifier les questions :**")
    modified_questions = st.text_area("Modifiez ou ajoutez des questions (une par ligne) :", "\n".join(questions))
    modified_questions_list = modified_questions.split("\n")

    if st.button("Analyser"):
        analyses = []

        with st.spinner('Analyse en cours...'):
            for question in modified_questions_list:
                if question.strip():
                    reponse = obtenir_reponse(question)
                    analyse = analyser_reponse(reponse, marque)
                    analyses.append((question, analyse))

        st.success("Analyse terminée !")

        # Synthèse globale
        top_marques = synthese_marques([a for q, a in analyses])
        st.write("**Synthèse des marques mentionnées :**")
        st.bar_chart(top_marques)

        # Synthèse des éléments sémantiques
        top_elements = synthese_elements_semantiques([a for q, a in analyses])
        st.write("**Synthèse des éléments sémantiques les plus mentionnés :**")
        st.bar_chart(top_elements)

        # Synthèse des sentiments globale
        sentiments = comparer_sentiments([a for q, a in analyses])
        st.write("**Synthèse globale des sentiments :**")
        st.bar_chart(sentiments)

        # Affichage des analyses détaillées
        st.write("**Détails des analyses :**")
        for i, (question, analyse) in enumerate(analyses):
            st.write(f"**Analyse de la question {i+1} :** {question}")
            st.write(f"- Marque mentionnée : {'Oui' if analyse['mention_marque'] else 'Non'}")
            st.write(f"- Sentiment : {analyse['sentiment']}")
            st.write(f"- Marques mentionnées : {', '.join(analyse['marques_mentionnees'])}")
            st.write(f"- Éléments sémantiques : {', '.join(analyse['elements_semantiques'])}")
