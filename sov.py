import streamlit as st
import openai
from transformers import pipeline

# Charger les secrets depuis le fichier secrets.toml
openai_key = st.secrets["openai_key"]

# Configuration de l'API OpenAI
openai.api_key = openai_key

# Charger le modèle de sentiment
sentiment_model = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

def obtenir_reponse(question):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message['content']

def extraire_marques(texte):
    prompt_marques = f"""Identifie les marques dans ce texte et donne moi la liste sous ce format : ["marque1", "marque2"].
    Texte : {texte}"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_marques}],
        temperature=0
    )
    return eval(response.choices[0].message['content'])

def extraire_elements_semantiques(texte):
    prompt_elements = f"""Identifie les éléments sémantiques importants dans ce texte et donne moi la liste sous ce format : ["prix", "taille", "qualité"].
    Texte : {texte}"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_elements}],
        temperature=0
    )
    return eval(response.choices[0].message['content'])

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

# Interface Streamlit
st.title("Analyse des Réponses des LLM")

questions = st.text_area("Entrez vos questions (une par ligne) :")
marque = st.text_input("Entrez la marque à analyser :")

if st.button("Analyser"):
    questions_list = questions.split('\n')
    for question in questions_list:
        if question.strip():
            st.write(f"**Question :** {question}")
            reponse = obtenir_reponse(question)
            st.write(f"**Réponse :** {reponse}")
            analyse = analyser_reponse(reponse, marque)
            st.write("**Analyse :**")
            st.write(f"- Marque mentionnée : {analyse['mention_marque']}")
            st.write(f"- Sentiment : {analyse['sentiment']}")
            st.write(f"- Marques mentionnées : {', '.join(analyse['marques_mentionnees'])}")
            st.write(f"- Éléments sémantiques : {', '.join(analyse['elements_semantiques'])}")
