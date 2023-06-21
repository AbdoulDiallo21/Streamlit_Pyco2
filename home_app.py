import streamlit as st
import streamlit.components.v1 as stc

#Ajout html
html_titre = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Application</h1>
		<h2 style="color:white;text-align:center;">Prédiction des émission de CO2 des véhicules immatriculés en 2019 en Europe </h2>
		</div>
		"""
html_descr = """
		<div style="background-color:white;padding:10px;border-radius:10px">
		<p style="color:black;text-align:left;">PyCo2 est un modèle de Machine Learning capable de prédire le Co2 émis par les véhicules, construit grâce à l’étude
    de bases de données techniques et caractéristiques de millions de véhicules.
    Notre démarche globale pour mener ce projet, ainsi qu’un module interactif pour tester notre algorithme seront présentés sur
    ce Streamlit).
        </p>
		</div>
		"""


#Structure de l'application

def run_home_app():
    #stc.html(html_titre)
   # st.subheader("Home")
    st.markdown("### Description du projet")
    st.markdown(html_descr, unsafe_allow_html=True)
    st.markdown("### Vous pouvez sélectionner sur le menu de gauche :")

    st.markdown("**1-Analyse exploratoire des données** : Présentation et analyse exploratoire du jeu de données exploité")
    st.markdown("""
        >> 1.1 **Présentation des données** 

        >> 1.2- **Statistiques et Visualisation dynamique**
        """)
    st.markdown("**3-Modélisation**: Prédiction, évaluation et interprétabilité")
    st.markdown("""
        **A vous de jouer:**
        >> **3.1- Valeurs:** vous choisissez les caractéristiques de votre vehicule et vous cliquez sur predit pour obtenir la prédiction de CO2 selon les trois modèles
        
        >> **3.2- Load data:** vous chargez un jeu de données et vous utiliser notre modèle pour calculer les émissions de CO2
        """)

    
    
   
    
