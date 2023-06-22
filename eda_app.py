# Importation des packages
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import io

#Charger les données
def load_data(data):
    df=pd.read_csv(data, sep=",")
    return df
#@st.cache
def run_eda_app():
    df = pd.read_csv('data_eda.csv', sep = ',',decimal=",", dtype={'index':'str'}, encoding='utf-8')
    for col in df.columns[0:10]:
        df[col]=df[col].astype(str)
    for col in df.columns[10:18]:
        df[col]=round(df[col].astype(float),2)
    df=df.rename(columns={'mck':'marque','cr_m1g':'v4x4'})
    df=df.copy()
    df['nom_carburant']=df['carbu']
    vcarbu=(2.0,3.0,4.0,5.0,6.0,7.0,8.0)
    ncarbu=('NG-NGBM-HDG','LPG','ESSENCE/ELECTRIC','DIESEL/ELECTRIC','SUPERETHANOL-E85','ESSENCE','DIESEL')
    df['nom_carburant']=df['nom_carburant'].replace(vcarbu,ncarbu)
    #Ajout de la variable
    df['all']='all'
    
    #df=load_data("data/dfco2_eu_2019.csv")
    #Ajout sous menu
    submenu = st.sidebar.selectbox("Exploration", ["Présentation des données","Statistiques et Visualisation"])
    html_descr_table ="""
        <p>Ce jeu de données contient les émissions de CO2 des véhicules imatriculés en europe en 2019 ainsi les caractéristiques techniques.Il est téléchargeable sur le site d'European Envronnment Agency depuis ce lien
        <a href="https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-20", target="_blank">https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-20</a></p>
        """
    if submenu == "Présentation des données":
        st.markdown("### Présentation du jeu de données")
        st.markdown(html_descr_table,unsafe_allow_html=True)
        if st.checkbox("Afficher le dictionnaire des variables"):
            st.markdown(
            """
            |Nom variable (initial)|Nom variable (récodé)|Libellé|Type|Pourcentage de données manquantes|
            | --- | --- | --- |--- |--- |
            |ID (clé primaire)|idve|Identifiant | object | 0|
            |Country| country|Pays| object| 0|
            |VFN|vfn|Numéro d'identification de la famille du véhicule| object| 10|
            |Mp|mp|Mutualisation des constructeurs|object | 3|
            |Mh|mh|Nom du Fabricant Dénomination standard de l'UE| object| 0|
            |Man|man|Nom du fabricant Déclaration OEM| object| 0|
            |MMS|mms|Nom du Fabricant Dénomination du registre MS| float| 100|
            |Tan|tan|Numéro d'homologation de type | object| 0.24|
            |T|tipe|Type| float| 0.03|
            |Va|variant|Variant| float| 0.18|
            |Ve|version|Version| float| 0.47|
            |Mk|mk|Constructeur|float | 0.002|
            |Cn|cn|Nom commercial| float| 0.13|
            |Ct|ct|Catégorie du type de véhicule réceptionné| float| 0.20|
            |Cr|cr|Catégorie du véhicule immatriculé| float| 0|
            |r|tisncr|Total des nouvelles inscriptions|int64 | 0|
            |m (kg)|mskg|Masse en ordre de marche (Véhicule terminé/complet)| float| 0.0003|
            |Mt|mt|Masse d'essai WLTP| float| 19|
            |Enedc (g/km)|enedcgkm|Émissions spécifiques de CO 2 (NEDC)| float| 75|
            |Ewltp (g/km)|ewltpgkm|Émissions spécifiques de CO 2 (WLTP)| float|0.84 |
            |W (mm)|wmm|largeur (Wheel Base)| float| 0.17|
            |At1 (mm)|at1mm|Largeur d'essieu essieu directeur(Axle width steering axle)| float| 15|
            |At2 (mm)|at2mm|Largeur d'essieu autre essieu(Axle width other axle)| float| 15|
            |Ft|ftcarb|Type de carburant| object|0 |
            |Fm|fmcarb|Mode carburant| object| 0.01| 
            |ec (cm3)|eccm3|Capacité moteur| float| 10|
            |ep (KW)|epkw|Puissance du moteur| float|0.89 |
            |z (Wh/km)|zkwkm|Consommation d'énergie électrique | float| 83|
            |IT|nitech|Inoovation technologique ou groupe d'innovations technologiques| object| 54|
            |Ernedc (g/km)|ernedcgkm|Réduction des émissions grâce à des technologies innovantes| floatt| 100|
            |Erwltp (g/km)|erwltpgkm|Réduction des émissions grâce à des technologies innovantes (WLTP)| float|56 |
            |De|fdev|Facteur de déviation| float| 100| 
            |Vf|vf|Facteur de vérification| float| 100|
            |Status|Status| P=Données provisoires, F = Données définitives| object|0 |
            |year|year|Année de déclaration| float| 0|
            |Date of registration|dmj| date de mise à jour|int64|10|
            |Fuel consumption|Consommation de fuel| consofuel| float| 40|
            |Electric range (km)| Gamme électrique|electricrge|float|85|
            """
            )
    if submenu == "Statistiques et Visualisation":
        if st.checkbox("Exploration des données"):
            st.markdown("**Affichage des cinq premières observations**")
            st.dataframe(df.head())
            st.markdown("**Affichage des cinq dernières observations**")
            st.dataframe(df.tail())
            col1, col2=st.columns([1,2])
            with col1:
                st.markdown("**Nombre de valeurs uniques par variable**")
                st.dataframe(df.nunique())
            with col2:
                buffer = io.StringIO()
                df.info(buf=buffer)
                infos= buffer.getvalue()
                st.markdown("**Informations sur les données**")
                st.text(infos)    
            st.markdown("**Résumé statistique des variables numériques**")
            st.dataframe(df.select_dtypes(include=float).describe())
            st.markdown("**Résumé statistique des variables catégorielles**")
            st.dataframe(df.select_dtypes(include=object).describe())
        st.markdown("***")
        st.markdown("### Sélectionner les variables à croiser:")
        if st.checkbox("Construire vos graphiques"):
                st.text("ici, vous pouvez construire les graphiques que vous souhaitez afficher")
                st.warning("🔥, Vous devez renseigner tous les champs pour la construction des graphiques")
                selected_var = st.selectbox('Choisir la variable qualitative',['all','marque', 'country', 'nom_carburant','turbo','sport','hybrid','coupe','reductco2it','minibus','break','v4x4'])
                selections_mod=list(df[selected_var].unique())
                selected_z_var=st.multiselect("Entrez vos sélections",selections_mod, selections_mod[0])
                df_select=df[df[selected_var].isin(selected_z_var)]
                selected_x_var = st.selectbox('Choisir la variable x',list(df.columns[10:18]))
                selected_y_var = st.selectbox('Choisir la variable y', list(df.columns[10:18]))
                if st.checkbox("Afficher le nuage de points"):
                    alt_chart = (
                    alt.Chart(df_select,title=f"Nuage de points de {selected_x_var} et {selected_y_var} selon {selected_var}")
                    .mark_point().encode(x=selected_x_var, y=selected_y_var, color=selected_var)
                    )
                    st.altair_chart(alt_chart, use_container_width=True)
                    
                if st.checkbox("Afficher la distribution de x et y"):
                    col1, col2=st.columns(2)
                    with col1:
                        alt_chart = (
                        alt.Chart(df_select).mark_bar().encode(
                        alt.X(selected_x_var),y='count()')
                        )
                        st.altair_chart(alt_chart, use_container_width=True)

                    with col2:
                        alt_chart = (
                        alt.Chart(df_select).mark_bar().encode(
                        alt.X(selected_y_var),y='count()')
                        )
                        st.altair_chart(alt_chart, use_container_width=True)
                
                if st.checkbox("Afficher le boxplot"):
                    alt_chart = (
                    alt.Chart(df_select).mark_boxplot(size=50, extent=1.5)
                    .encode(y=alt.Y(selected_y_var),x=alt.X(selected_var),color=selected_var)
                    .interactive()
                    .properties(width=300).configure_axis(
                        labelFontSize=16,
                        titleFontSize=16)
                    )
                    st.altair_chart(alt_chart, use_container_width=True)
                
                if st.checkbox("Afficher le graphique de la variable qualitative"):
                    dfquali=df.groupby([selected_var], as_index=False).size().rename(columns={'size':'n'})
                    dfquali['pct']=round(100*dfquali['n']/df.shape[0],2)

                    alt_chart = (
                        alt.Chart(dfquali, title=f"Répartition en % du nombre de vehicules par {selected_var}").mark_arc().encode(
                        theta='pct', color=selected_var)
                    )
                    st.altair_chart(alt_chart, use_container_width=True)







                                          
     
            




        




