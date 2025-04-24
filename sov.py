import streamlit as st
from openai import OpenAI
from transformers import pipeline
import ast
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd

# Charger les secrets depuis le fichier secrets.toml
openai_key = st.secrets["openai_key"]

# Configuration de l'API OpenAI
client = OpenAI(api_key=openai_key)

# Charger le modèle de sentiment
sentiment_model = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

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
    prompt_marques = f"""
Tu es un assistant qui identifie les marques dans un texte. Extrait uniquement les noms de marques connus (entreprises, produits, etc.) sous forme de liste Python.

Format attendu : ["Nike", "Apple", "Samsung"]

Texte :
\"\"\"{texte}\"\"\"
"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_marques}],
            temperature=0
        )
        content = completion.choices[0].message.content

        # Nettoyage : extraire la liste entre crochets s’il y a du texte autour
        match = re.search(r'\[(.*?)\]', content, re.DOTALL)
        if match:
            extrait = "[" + match.group(1) + "]"
            marques = ast.literal_eval(extrait)
            if isinstance(marques, list) and all(isinstance(m, str) for m in marques):
                return [m.strip() for m in marques if m.strip()]
    except Exception as e:
        st.warning(f"Erreur lors de l'extraction des marques : {e}")
    return []

def extraire_elements_semantiques(texte):
    prompt_elements = f"""Identifie les éléments sémantiques importants dans ce texte et donne-moi la liste sous ce format : ["prix", "taille", "qualité"].\nTexte : {texte}"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_elements}],
        temperature=0.2
    )
    try:
        return ast.literal_eval(completion.choices[0].message.content)
    except (SyntaxError, ValueError):
        return []

def analyser_reponse(reponse, marque):
    try:
        sentiment_result = sentiment_model(reponse)[0]
        sentiment = sentiment_result['label']
    except Exception as e:
        st.error(f"Erreur lors de l'analyse de sentiment : {e}")
        sentiment = "Erreur"

    marques_mentionnees = extraire_marques(reponse)
    elements_semantiques = extraire_elements_semantiques(reponse)
    mention_marque = marque.lower() in reponse.lower()

    return {
        "mention_marque": mention_marque,
        "sentiment": sentiment,
        "marques_mentionnees": marques_mentionnees,
        "elements_semantiques": elements_semantiques
    }

def comparer_sentiments(analyses):
    sentiments = {"Very Positive": 0, "Positive": 0, "Neutral": 0, "Negative": 0, "Very Negative": 0, "Erreur": 0}
    for analyse in analyses:
        sentiments[analyse["sentiment"]] += 1
    return sentiments

def synthese_marques(analyses):
    marques_count = {}
    for analyse in analyses:
        for marque_mentionnee in analyse["marques_mentionnees"]:
            marques_count[marque_mentionnee] = marques_count.get(marque_mentionnee, 0) + 1
    top_marques = sorted(marques_count.items(), key=lambda x: x[1], reverse=True)[:9]
    return dict(top_marques)

def synthese_elements_semantiques(analyses):
    elements_count = {}
    for analyse in analyses:
        for element in analyse["elements_semantiques"]:
            elements_count[element] = elements_count.get(element, 0) + 1
    top_elements = sorted(elements_count.items(), key=lambda x: x[1], reverse=True)[:10]
    return dict(top_elements)

def classifier_intention(question):
    prompt_intention = f"""Classifie la question suivante dans l'une des catégories suivantes : [Recherche], [Support], [Comparaison], [Achat].\nQuestion : {question}"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_intention}],
        temperature=0
    )
    output = completion.choices[0].message.content.strip()
    match = re.search(r"\[(.*?)\]", output)
    return match.group(1) if match else "Inconnue"

def analyse_avis(marque):
    prompt_avis = f"""Donne un avis sur la marque : {marque} avec une répartition entre avantages et inconvénients. Cours et précis."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_avis}],
        temperature=0
    )
    return completion.choices[0].message.content

# Interface Streamlit
st.image("SlayLLM.jpg", width=200)
st.title("Analyse des Réponses des LLM")

mode = st.radio("Choisissez le mode d'entrée :", ("Entrer le nom d'une marque", "Entrer manuellement une liste de questions"))

if mode == "Entrer le nom d'une marque":
    marque = st.text_input("Entrez la marque à analyser :")
    if marque:
        questions = generer_questions(marque)
        st.write("**Questions générées :**")
        for question in questions:
            st.write(f"- {question}")

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

            top_marques = synthese_marques([a for q, a in analyses])
            st.write("**Synthèse des marques mentionnées :**")
            st.bar_chart(top_marques)

            top_elements = synthese_elements_semantiques([a for q, a in analyses])
            st.write("**Synthèse des éléments sémantiques les plus mentionnés :**")
            df_elements = pd.DataFrame(list(top_elements.items()), columns=["Élément", "Occurrences"])
            st.dataframe(df_elements)

            sentiments = comparer_sentiments([a for q, a in analyses])
            st.write("**Synthèse globale des sentiments :**")
            st.bar_chart(sentiments)

            total_mentions = sum(top_marques.values())
            market_share = {marque: (count / total_mentions) * 100 for marque, count in top_marques.items()}
            st.write("**Market Share :**")
            st.bar_chart(market_share)

            labels = list(market_share.keys())
            sizes = list(market_share.values())
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title("Répartition des mentions des marques")
            st.pyplot(fig)

            intentions = {"Recherche": 0, "Support": 0, "Comparaison": 0, "Achat": 0, "Inconnue": 0}
            for question, _ in analyses:
                intention = classifier_intention(question)
                intentions[intention] += 1
            st.write("**Distribution des intentions :**")
            st.bar_chart(intentions)

            avis = analyse_avis(marque)
            st.write("**Analyse des avis :**")
            st.write(avis)

            st.write("**Détails des analyses :**")
            for i, (question, analyse) in enumerate(analyses):
                st.write(f"**Analyse de la question {i+1} :** {question}")
                st.write(f"- Marque mentionnée : {'Oui' if analyse['mention_marque'] else 'Non'}")
                st.write(f"- Sentiment : {analyse['sentiment']}")
                st.write(f"- Marques mentionnées : {', '.join(analyse['marques_mentionnees'])}")
                st.write(f"- Éléments sémantiques : {', '.join(analyse['elements_semantiques'])}")

elif mode == "Entrer manuellement une liste de questions":
    st.write("**Entrez vos questions (une par ligne) :**")
    questions = st.text_area("Questions :")
    questions_list = questions.split("\n")

    if st.button("Analyser"):
        analyses = []

        with st.spinner('Analyse en cours...'):
            for question in questions_list:
                if question.strip():
                    reponse = obtenir_reponse(question)
                    analyse = analyser_reponse(reponse, "")
                    analyses.append((question, analyse))

        st.success("Analyse terminée !")

        top_marques = synthese_marques([a for q, a in analyses])
        st.write("**Synthèse des marques mentionnées :**")
        st.bar_chart(top_marques)

        top_elements = synthese_elements_semantiques([a for q, a in analyses])
        st.write("**Synthèse des éléments sémantiques les plus mentionnés :**")
        st.bar_chart(top_elements)

        sentiments = comparer_sentiments([a for q, a in analyses])
        st.write("**Synthèse globale des sentiments :**")
        st.bar_chart(sentiments)

        total_mentions = sum(top_marques.values())
        market_share = {marque: (count / total_mentions) * 100 for marque, count in top_marques.items()}
        st.write("**Market Share :**")
        st.bar_chart(market_share)

        labels = list(market_share.keys())
        sizes = list(market_share.values())
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title("Répartition des mentions des marques")
        st.pyplot(fig)

        intentions = {"Recherche": 0, "Support": 0, "Comparaison": 0, "Achat": 0, "Inconnue": 0}
        for question, _ in analyses:
            intention = classifier_intention(question)
            intentions[intention] += 1
        st.write("**Distribution des intentions :**")
        st.bar_chart(intentions)

        st.write("**Détails des analyses :**")
        for i, (question, analyse) in enumerate(analyses):
            st.write(f"**Analyse de la question {i+1} :** {question}")
            st.write(f"- Marque mentionnée : {'Oui' if analyse['mention_marque'] else 'Non'}")
            st.write(f"- Sentiment : {analyse['sentiment']}")
            st.write(f"- Marques mentionnées : {', '.join(analyse['marques_mentionnees'])}")
            st.write(f"- Éléments sémantiques : {', '.join(analyse['elements_semantiques'])}")
