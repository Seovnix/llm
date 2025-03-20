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
    sentiment_result = sentiment_model(reponse)[0]
    sentiment = sentiment_result['label']
    marques_mentionnees = extraire_marques(reponse)
    elements_semantiques = extraire_elements_semantiques(reponse)
    mention_marque = marque.lower() in reponse.lower()
    return {
        "mention_marque": mention_marque,
        "sentiment": sentiment,
        "marques_mentionnees": marques_mentionnees,
        "elements_semantiques": elements_semantiques
    }

def comparer_sentiments_par_marque(analyses):
    sentiments_par_marque = {}
    for analyse in analyses:
        for marque in analyse["marques_mentionnees"]:
            if marque not in sentiments_par_marque:
                sentiments_par_marque[marque] = {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0}
            sentiments_par_marque[marque][analyse["sentiment"]] += 1
    return sentiments_par_marque

def synthese_marques(analyses):
    marques_count = {}
    for analyse in analyses:
        for marque_mentionnee in analyse["marques_mentionnees"]:
            marques_count[marque_mentionnee] = marques_count.get(marque_mentionnee, 0) + 1
    return dict(sorted(marques_count.items(), key=lambda x: x[1], reverse=True)[:5])

def synthese_elements_semantiques(analyses):
    elements_count = {}
    for analyse in analyses:
        for element in analyse["elements_semantiques"]:
            elements_count[element] = elements_count.get(element, 0) + 1
    return dict(sorted(elements_count.items(), key=lambda x: x[1], reverse=True)[:5])

# Interface Streamlit
st.image("GRM-Nexus-16_9.png", width=500)
st.title("Analyse des Réponses des LLM")

questions = st.text_area("Entrez vos questions (une par ligne) :")
marque = st.text_input("Entrez la marque à analyser :")

if st.button("Analyser"):
    questions_list = [q.strip() for q in questions.split('\n') if q.strip()]
    analyses = []

    with st.spinner('Analyse en cours...'):
        for question in questions_list:
            reponse = obtenir_reponse(question)
            analyse = analyser_reponse(reponse, marque)
            analyses.append((question, analyse))

    st.success("Analyse terminée !")

    # Synthèse des marques
    top_marques = synthese_marques([a for _, a in analyses])
    st.write("**Synthèse des marques mentionnées :**")
    if top_marques:
        st.bar_chart(top_marques)
    else:
        st.write("Aucune marque mentionnée.")

    # Synthèse des éléments sémantiques
    top_elements_semantiques = synthese_elements_semantiques([a for _, a in analyses])
    st.write("**Synthèse des éléments sémantiques les plus mentionnés :**")
    if top_elements_semantiques:
        st.bar_chart(top_elements_semantiques)
    else:
        st.write("Aucun élément sémantique mentionné.")

    # Comparaison des sentiments par marque
    sentiments_par_marque = comparer_sentiments_par_marque([a for _, a in analyses])
    st.write("**Synthèse du sentiment pour chaque marque :**")
    for marque, sentiments in sentiments_par_marque.items():
        st.write(f"**{marque}**")
        fig, ax = plt.subplots()
        ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=['darkred', 'red', 'gray', 'lightgreen', 'green'])
        ax.set_title(f"Sentiments pour {marque}")
        st.pyplot(fig)

    # Affichage des analyses détaillées
    st.write("**Détails des analyses :**")
    for i, (question, analyse) in enumerate(analyses):
        st.write(f"**Analyse de la question {i+1} :** {question}")
        st.write(f"- Marque mentionnée : {'Oui' if analyse['mention_marque'] else 'Non'}")
        st.write(f"- Sentiment : {analyse['sentiment']}")
        st.write(f"- Marques mentionnées : {', '.join(analyse['marques_mentionnees'])}")
        st.write(f"- Éléments sémantiques : {', '.join(analyse['elements_semantiques'])}")
