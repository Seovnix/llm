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

# Charger le modèle de sentiment
sentiment_model = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

def obtenir_reponse(question):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )
    return completion.choices[0].message.content

def extraire_marques(texte):
    prompt_marques = f"""Identifie les marques dans ce texte et donne moi la liste sous ce format : ["marque1", "marque2"].
    Texte : {texte}"""
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_marques}],
        temperature=0
    )
    try:
        return ast.literal_eval(completion.choices[0].message.content)
    except (SyntaxError, ValueError):
        return []

def extraire_elements_semantiques(texte):
    prompt_elements = f"""Identifie les éléments sémantiques importants dans ce texte et donne moi la liste sous ce format : ["prix", "taille", "qualité"].
    Texte : {texte}"""
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_elements}],
        temperature=0
    )
    try:
        return ast.literal_eval(completion.choices[0].message.content)
    except (SyntaxError, ValueError):
        return []

def analyser_reponse(reponse, marque):
    # Analyse de sentiment
    sentiment_result = sentiment_model(reponse)[0]
    sentiment = sentiment_result['label']

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

def comparer_sentiments(analyses, marque):
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for analyse in analyses:
        if marque.lower() in analyse["marques_mentionnees"]:
            if analyse["sentiment"] == "positive":
                sentiments["positive"] += 1
            elif analyse["sentiment"] == "neutral":
                sentiments["neutral"] += 1
            else:
                sentiments["negative"] += 1
    return sentiments

def synthese_marques(analyses, marque):
    marques_count = {}
    for analyse in analyses:
        for marque_mentionnee in analyse["marques_mentionnees"]:
            if marque_mentionnee in marques_count:
                marques_count[marque_mentionnee] += 1
            else:
                marques_count[marque_mentionnee] = 1

    # Garder les 5 marques les plus mentionnées
    top_marques = sorted(marques_count.items(), key=lambda x: x[1], reverse=True)[:5]
    return {marque: count for marque, count in top_marques}

# Interface Streamlit
st.image("https://www.groupm.com/themes/custom/groupm/logo.svg", width=200)
st.title("Analyse des Réponses des LLM")

questions = st.text_area("Entrez vos questions (une par ligne) :")
marque = st.text_input("Entrez la marque à analyser :")

if st.button("Analyser"):
    questions_list = questions.split('\n')
    analyses = []

    with st.spinner('Analyse en cours...'):
        for question in questions_list:
            if question.strip():
                reponse = obtenir_reponse(question)
                analyse = analyser_reponse(reponse, marque)
                analyses.append((question, analyse))

    st.success("Analyse terminée !")

    # Synthèse globale
    top_marques = synthese_marques([a for q, a in analyses], marque)
    st.write("**Synthèse des marques mentionnées :**")
    st.bar_chart(top_marques)

    # Comparaison des sentiments
    sentiments = comparer_sentiments([a for q, a in analyses], marque)
    st.write("**Comparaison des sentiments pour votre marque :**")

    # Vérification des valeurs NaN
    if any(np.isnan(list(sentiments.values()))):
        st.error("Erreur : Les données de sentiment contiennent des valeurs manquantes.")
    else:
        # Graphique en camembert pour les sentiments
        fig, ax = plt.subplots()
        ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=['green', 'gray', 'red'])
        ax.set_title("Répartition des sentiments pour votre marque")
        st.pyplot(fig)

    # Affichage des analyses détaillées
    st.write("**Détails des analyses :**")
    for i, (question, analyse) in enumerate(analyses):
        st.write(f"**Analyse de la question {i+1} :** {question}")
        st.write(f"- Marque mentionnée : {'Oui' if analyse['mention_marque'] else 'Non'}")
        st.write(f"- Sentiment : {analyse['sentiment']}")
        st.write(f"- Marques mentionnées : {', '.join(analyse['marques_mentionnees'])}")
        st.write(f"- Éléments sémantiques : {', '.join(analyse['elements_semantiques'])}")
