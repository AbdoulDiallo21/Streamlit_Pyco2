# Importation des packages
import streamlit as st
import streamlit.components.v1 as stc
#Import autres applications
from home_app import run_home_app
from eda_app import run_eda_app
from ml_app import run_ml_app

#Structure de l'application
def main():
    #stc.html(html_titre)
    menu=["Home page","Analyse exploratoire des données","Modelisation", "About"]
    st.sidebar.image("logoPyCo2.png", use_column_width=True, width=400)
    choice=st.sidebar.selectbox("Menu",menu)
    #Ajout du logo pour les pages en le centrant
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.write("")
    # with col2:
    #     st.image("logoPyCo2.png")
    # with col3:
    #     st.write("")
    st.markdown("***")
    st.title(":blue[Application de prévision du taux de $CO_2$ émis par les véhicules]")
    st.markdown("***")
    if choice=="Home page":
        run_home_app()
    elif choice=="Analyse exploratoire des données":
        run_eda_app()
    elif choice=="Modelisation":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("")
        with col2:
            st.image("co2.jpg")
        with col3:
            st.write("")
        run_ml_app()
    else:
        st.subheader("About")

if __name__=='__main__':
    main()
