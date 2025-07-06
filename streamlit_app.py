# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Segmentation Magasins", layout="wide")

# Titre
st.title("Segmentation des magasins avec K-Means")

# Importation des données
uploaded_file = st.file_uploader("Importer le fichier Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Colonnes détectées :", df.columns.tolist())


    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    # Sélection des variables
    features = ['Chiffre d’affaires', 'Clients/jour', 'Surface', 'Employés']
    X = df[features]

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Facultatif : afficher un extrait des données standardisées
    st.subheader("📐 Données standardisées (extrait)")
    st.dataframe(pd.DataFrame(X_scaled, columns=features).head())


    # Méthode du coude
    st.subheader("Méthode du coude")
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_range, inertias, 'bo-')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Inertie')
    ax.set_title("Méthode du coude")
    st.pyplot(fig)

    # Choix de K
    k = st.slider("Choisir le nombre de clusters (K)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    st.subheader("Données segmentées")
    st.dataframe(df)

    # Moyenne par cluster
    st.subheader("Moyennes par cluster")
    st.dataframe(df.groupby('Cluster')[features].mean())

    # Visualisation 2D
    st.subheader("Visualisation des clusters (CA vs Clients/jour)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='Chiffre d’affaires', y='Clients/jour', hue='Cluster', data=df, palette="tab10", ax=ax2)
    st.pyplot(fig2)
