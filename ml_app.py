# Importation des packages
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import altair 
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
#packages pour la regression lineaire
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate
#packages métriques de contrôle performance du modèle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import joblib
from joblib import dump, load
import shap
from streamlit_shap import st_shap
from streamlit_echarts import st_echarts

def run_ml_app():
    st.subheader("Modèles")
    st.markdown("**Test des meilleurs modèles**")
    st.markdown(""""
    Nous n'avaons pas pu charger notre meilleur modèle Knn regressor car il est impossible de le charger sur github compte
    tenu de taille qui dépasse 25MB. Cependant les modèles déployés ici sont très proches en terme de perfomance à celui de Knn.
    """)
    add_selectbox = st.sidebar.selectbox("type",("Valeurs", "load data"))
    # Chargement de chaque modèle et prédiction des émissions de co2 sur l'ensemble des données
    best_rforestreg=load('best_rfreg.joblib')
    best_gbreg=load('best_gbreg.joblib')
    #best_knnreg=load('best_knn_reg.joblib')
    
    if add_selectbox == 'Valeurs':
        col1, col2 = st.columns(2)
        fueltype = col1.selectbox('Choisir le type de carburant', ['NG','NGBM','HDG','LPG','ESSENCE\ELECTRIC','DIESEL\ELECTRIC','SUPERETHANOL-E85','ESSENCE','DIESEL'])
        consofuel = col2.slider('Consommation de fuel (litre / 100 km)', 0, 20, 6, step=1)
        epkw = col1.slider('Puissance électrique en KW', 0, 1000,100, step=10)
        eccm3 = col2.slider('Capacité du moteur', 800, 7000, 2000, step=100)
        mskg = col1.slider('Masse en ordre de marche en kg', 650, 3500, 1000,step=10)
        at1mm = col2.slider('Largeur de voie en millimètre', 100, 3000, 500,step=100)
        electricrge = col1.slider('Autonomie électrique en KM', min_value = 0.0, max_value = 500.0)

        input_dict = {'fueltype' : fueltype, 'consofuel' : consofuel, 'electricrge':electricrge,'mskg' : mskg, 'at1mm' : at1mm, 'eccm3' : eccm3, 'epkw' : epkw}
        input_df = pd.DataFrame([input_dict])
        
        #Convertir le type de fuel en valeur numérique
        def valeur_carbu(x):
            if x in ['NG','NGBM','HDG']:
                return 2
            elif x =='LPG':
                return 3
            elif x=='ESSENCE\ELECTRIC':
                return 4
            elif x=='DIESEL\ELECTRIC':
                return 5
            elif x=='SUPERETHANOL-E85':
                return 6
            elif x=='ESSENCE':
                return 7
            elif x=='DIESEL':
                return 8
        input_df['carbu']=input_df['fueltype'].apply(valeur_carbu)
        input_df['carbu']=input_df['carbu'].astype(float)
        input_df=input_df[['mskg', 'at1mm', 'carbu', 'eccm3', 'epkw', 'consofuel', 'electricrge']]
        st.write("Vos données")
        st.dataframe(input_df)
      
        #Standarisation des features
        # Valeurs moyennes des features 
        input_df['mean_carbu']=6.86
        input_df['mean_consofuel']=5.29
        input_df['mean_electricrge']=4.76
        input_df['mean_mskg']=1455.13
        input_df['mean_at1mm']=1548.04
        input_df['mean_eccm3']=1516.20
        input_df['mean_epkw']=102.43
        # Ecart-type des features
        input_df['ec_carbu']=1.25
        input_df['ec_consofuel']=1.58
        input_df['ec_electricrge']=16.36
        input_df['ec_mskg']=314.39
        input_df['ec_at1mm']=60.12
        input_df['ec_eccm3']=504.08
        input_df['ec_epkw']=47.75

        # Standarization
        input_df['carbu_scaled']=(input_df['carbu']-input_df['mean_carbu'])/input_df['ec_carbu']
        input_df['consofuel_scaled']=(input_df['consofuel']-input_df['mean_consofuel'])/input_df['ec_consofuel']
        input_df['electricrge_scaled']=(input_df['electricrge']-input_df['mean_electricrge'])/input_df['ec_electricrge']
        input_df['mskg_scaled']=(input_df['electricrge']-input_df['mean_mskg'])/input_df['ec_mskg']
        input_df['at1mm_scaled']=(input_df['at1mm']-input_df['mean_at1mm'])/input_df['ec_at1mm']
        input_df['eccm3_scaled']=(input_df['eccm3']-input_df['mean_eccm3'])/input_df['ec_eccm3']
        input_df['epkw_scaled']=(input_df['eccm3']-input_df['mean_epkw'])/input_df['ec_epkw']

        # Features
        feats=input_df[['carbu_scaled','consofuel_scaled','electricrge_scaled','mskg_scaled','at1mm_scaled','eccm3_scaled','epkw_scaled']]

        feats=feats.rename(columns={'carbu_scaled':'carbu' , 'consofuel_scaled' : 'consofuel', 'electricrge_scaled':'electricrge',
                                    'mskg_scaled' : 'mskg', 'at1mm_scaled' : 'at1mm', 'eccm3_scaled' : 'eccm3', 'epkw_scaled' : 'epkw'})
        
        #st.dataframe(feats)
       
        #Ajout Critère label aux prédictions
        def eti(x):
            if 0 < x <= 100:
                return 'A'
            elif 101 <= x <= 120:
                return 'B'
            elif 121 <= x <= 140:
                return 'C'
            elif 141 <= x <= 160:
                return 'D'
            elif 161 <= x <= 200:
                return 'E'
            elif 201 <= x <= 250:
                return 'F'
            else :
                return 'G'
            
        if st.button("Predict"):
            ypred_rfr= best_rforestreg.predict(feats)
            ypred_gbr= best_gbreg.predict(feats)
            #ypred_knn= best_knnreg.predict(feats)
            #output_dict={'RFR':ypred_rfr,'GBR':ypred_gbr,'KNN':ypred_knn}
            output_dict={'RFR':ypred_rfr,'GBR':ypred_gbr}
            df_output=pd.DataFrame(output_dict)
            df_output=pd.melt(df_output, var_name='Models',value_name='predictions')
            df_output.index=df_output['Models']
            df_output=df_output.drop('Models', axis=1)
            
            col1, col2= st.columns(2)
            with col1:
                df_output['VIGNETTE'] = df_output['predictions'].apply(eti)
                st.markdown("**Prédictions de CO2 par les trois modèles**")
                st.dataframe(df_output.style.highlight_min(axis=0))
                st.write("Dans ce tableau, les faibles prédictions sont en jaune.")
            with col2:
                st.markdown("**Vignette Ecolabel CO2**")
                st.image("vignette.jpg")

    if add_selectbox == "load data":
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            if data.shape[0]>10000:
                data=data.sample(n=10000)
            else:
                data
            #Selection des variables pour le modèle
            dataselect=data[['carbu','consofuel','electricrge','mskg','at1mm', 'eccm3', 'epkw']]
            datatarget=data[['ewltpgkm']]
            st.write(dataselect.shape)

            #Statistiques descriptives
            if st.checkbox("Statistiques descriptives des features"):
                st.write("Affichage des cinq premières observations")
                st.dataframe(dataselect.head(5))
                st.write("Affichage des cinq dernières observations")
                st.dataframe(dataselect.tail(5))
                st.subheader('Statistiques')
                st.write("Résumé statistique des variables")
                st.dataframe(dataselect.describe().T)
            st.write("---")
            #Standarization
            scaler=StandardScaler()
            feats = pd.DataFrame(scaler.fit_transform(dataselect),index=dataselect.index)
            feats.columns=dataselect.columns
            ypred_rfr= best_rforestreg.predict(feats)
            ypred_gbr= best_gbreg.predict(feats)
            #ypred_knn= best_knnreg.predict(feats)
            #output_dict2={'RFR':ypred_rfr,'GBR':ypred_gbr,'KNN':ypred_knn}
            output_dict2={'RFR':ypred_rfr,'GBR':ypred_gbr}
            predictions=pd.DataFrame(output_dict2)
            #Ajout Critère label aux prédictions
            def eti(x):
                if 0 < x <= 100:
                    return 'A'
                elif 101 <= x <= 120:
                    return 'B'
                elif 121 <= x <= 140:
                    return 'C'
                elif 141 <= x <= 160:
                    return 'D'
                elif 161 <= x <= 200:
                    return 'E'
                elif 201 <= x <= 250:
                    return 'F'
                else :
                    return 'G'
            #Application aux trois modèles
            st.markdown("#### Calcul de la prédiction")
            if st.button("Predict CO2"):
                col1, col2= st.columns(2)
                with col1:
                    preds1=pd.DataFrame(predictions.mean(), columns=['MEAN_CO2'])
                    preds1['VIGNETTE'] = preds1['MEAN_CO2'].apply(eti)
                    preds2=pd.DataFrame(predictions.std(), columns=['ECART_TYPE'])
                    preds=pd.concat([preds1,preds2], axis=1)
                    #Ajout ecart type ou histogramme
                    st.markdown("**Prédictions de CO2 par les trois modèles**")
                    st.dataframe(preds.style.highlight_min(axis=0))
                    st.write("Dans ce tableau, les valeurs faibles sont en jaune.")
                with col2:
                    st.markdown("**Vignette Ecolabel CO2**")
                    st.image("vignette.jpg")

            st.markdown("***")
            if st.checkbox("Taux de prediction"):
                st.write("Ici, le taux calculé nous permet de savoir si les modèles prédisent des valeurs proches.")

                predictions['RFR_VIGNETTE'] = predictions['RFR'].apply(eti)
                predictions['GBR_VIGNETTE'] = predictions['GBR'].apply(eti)
                #predictions['KNN_VIGNETTE'] = predictions['KNN'].apply(eti)

                t12=pd.crosstab(predictions['RFR_VIGNETTE'],predictions['GBR_VIGNETTE'],
                    rownames=['RFR'], colnames=['GBR'])
                t12b=pd.DataFrame(t12.stack()).reset_index()
                t12b.columns=['RFR','GBR','Nombre']

                def calcmetric(x,y,z):
                    if x==y:
                        return z
                    else:
                        return 0
                    
                t12b['nb_ident']=t12b[['RFR','GBR','Nombre']].apply(lambda x: calcmetric(*x), axis=1)
                tauxpred_rfr_gbr=t12b['nb_ident'].sum()/len(data)
                st.markdown("#### Comparaison des prédictions 2 à 2 des trois modèles")
                st.markdown(f"**RFR et GBR: :** {tauxpred_rfr_gbr}")
                
                # t13=pd.crosstab(predictions['RFR_VIGNETTE'],predictions['KNN_VIGNETTE'],
                #     rownames=['RFR'], colnames=['KNN'])
                # t13b=pd.DataFrame(t13.stack()).reset_index()
                # t13b.columns=['RFR','KNN','Nombre']
                # t13b['nb_ident']=t13b[['RFR','KNN','Nombre']].apply(lambda x: calcmetric(*x), axis=1)
                # tauxpred_rfr_knn=t13b['nb_ident'].sum()/len(data)
                # st.markdown(f"**RFR et KNN :** {tauxpred_rfr_knn}")
                # t23=pd.crosstab(predictions['GBR_VIGNETTE'],predictions['KNN_VIGNETTE'],
                #     rownames=['GBR'], colnames=['KNN'])
                # t23b=pd.DataFrame(t23.stack()).reset_index()
                # t23b.columns=['GBR','KNN','Nombre']
                # t23b['nb_ident']=t23b[['GBR','KNN','Nombre']].apply(lambda x: calcmetric(*x), axis=1)
                # tauxpred_gbr_knn=t23b['nb_ident'].sum()/len(data)
                # st.markdown(f"**GBR et KNN :** {tauxpred_gbr_knn}")
            st.markdown("***")
            if st.checkbox("Interpretabilité des modèles"):
                st.markdown("""
                Pour rappel, les modèles de ML sont souvent considéreées comme des boites noires (du plus au moins complexe) et afin de comprendre les résulats issus de ces modèles,
                nous avons ajouté ici une méthode d'interpretabilité. Nous avons fait le choix d'utiliser les méthodes de Shap (Shapley-Additve Explanations) pour tenter de mieux expliquer
                les résultats obtenus par nos modèles.
                """)
                with st.expander("Méthode Shap"):
                    st.write("""
                    Il convient de souligner que les valeurs SHAP sont fondées sur les valeurs de Shapley issues de la théorie
                    des jeux coopératifs, qui visent à répartir équitablement les gains entre les joueurs d’une coalition en
                    fonction de leur contribution à la valeur globale du jeu. Dans le cadre de l’apprentissage automatique,
                    les valeurs SHAP sont utilisées pour attribuer une valeur à chaque fonctionnalité dans un modèle, en
                    tenant compte de son impact sur la prédiction et de sa relation avec d’autres fonctionnalités. Si deux
                    fonctionnalités sont fortement corrélées, les valeurs SHAP permettent de déterminer la contribution de
                    chaque fonctionnalité de manière indépendante à la prédiction.

                    Les valeurs SHAP présentent deux avantages majeurs : elles fournissent une interprétation globale de la
                    contribution de chaque fonctionnalité à la prédiction (positive ou négative), ainsi qu’une interprétation
                    locale pour chaque observation, permettant de comprendre sa contribution à la prédiction.

                    En utilisant ces valeurs, nous pouvons évaluer la compréhensibilité de nos modèles et sélectionner les fonctionnalités
                    les plus importantes pour notre application.
                    """)
                st.markdown("---")
                explainer_rfr = shap.Explainer(best_rforestreg, feature_names=feats.columns)
                shap_values_rfr = explainer_rfr.shap_values(feats.values)
                explainer_gbr = shap.Explainer(best_gbreg, feature_names=feats.columns)
                shap_values_gbr = explainer_gbr.shap_values(feats.values)

                st.markdown("### Interpretabilité globale")
                st.markdown("#### Diagramme des valeurs moyennes de SHAP")
                st.write("""
                Ce graphique affiche les valeurs SHAP moyennes pour chaque feature, triées par ordre décroissant de leur importance.
                les variables en haut contribuent plus au modèle que ceux d'en bas, donc elles ont un pouvoir prédictif élévé.
                """)
                col1, col2, col3=st.columns([2,1,2])
                with col1:
                    st.markdown("**RandomForest Regressor (RFR)**")
                    st_shap(shap.summary_plot(shap_values_rfr,feats,plot_type="bar", color='green', plot_size=[6,4]))
                with col2:
                    st.write("--")
                with col3:
                    st.markdown("**Gradient Boosting Regressor (GBR)**")
                    st_shap(shap.summary_plot(shap_values_gbr, feats, plot_type="bar", plot_size=[6,4]))
                
                st.markdown("#### Diagramme des valeurs de SHAP")
                st.markdown("""
                **Ce graphique met en évidence**:
                 - les caractéristiques importantes et les effets sur l'ensemble des données.
                 - les relations positives et négatives des features avec la variable cible.
                **Comment lire ce graphique**:
                - Importance des caractéristiques: les features sont classés par ordre décroissant (du **+** au **-** important) 
                - (Impact) la ligne horizontale sépare en deux effets des features sur la cible: faible à gauche et forte à droite.
                - Couleur: le rouge indique les valeurs fortes et le bleu les faibles faibles.
                <br>
                **Exemple 1: Consommation de fuel**
                - Une faible valeur de consommation de fuel (en bleu) a un impact négatif négatif sur l'émission de CO2 et une forte valeur (en rouge) aura un impact positif.
                <br>
                **Exemple 2: Autonomie électrique**
                - Plus l'autonomie électrique augmente plus les émissions de CO2 deviennent faibles.

                """)
                col1, col2, col3=st.columns([2,1,2])
                with col1:
                    st.markdown("RandomForest Regressor (RFR)")
                    st_shap(shap.summary_plot(shap_values_rfr, feats, plot_size=[6,6]))
                    
                with col2:
                    st.write("--")
                with col3:
                    st.markdown("Gradient Boosting Regressor (GBR)")
                    st_shap(shap.summary_plot(shap_values_gbr, feats, plot_size=[6,6]))
                st.markdown("- Diagramme de dependance SHAP")
                st.markdown("""
                Le diagramme de dependance appelé encore diagramme de dépendance partielle montre l'effet marginal d'une ou deux features sur le résulat prédit
                d'un modèle. Il est montre si la relation entre la cible et une feature est linéiare, monotone ou complexe. Lorsqu'on regarde la dépendance partielle
                d'une variable, cela inclut toujours une autre variable avec laquelle elle interagit le plus.
                """)
                select_var_dep=st.selectbox("Choisir votre variable",list(feats.columns))
                col1, col2=st.columns([1,1])
                st.write("RandomForestRegressor")
                st_shap(shap.dependence_plot(select_var_dep, shap_values_rfr,feats))
                st.write("GradientBoostingRegressor")
                st_shap(shap.dependence_plot(select_var_dep, shap_values_gbr,feats))
                st.markdown("Inteprétabilité locale")
                st.markdown("""
                La fonction **shap_force** qui permet d'interpreter localement une observation, dispose de trois valeurs:
                - (1) la valeur de base: c'est la valeur moyenne qui serait prédite si aucune caractéristique n'est connue.
                - (2) la valeur SHAP ou output value est la valeur prédite par le modèle
                - (3) les valeurs SHAP de chaque variable propportionnelles aux tailles des flèches poussent la valeur prédite jusqu'à la valeur de base.
                     Celles qui sont en rouge poussent la prediction vers le haut et en bleu la poussent vers le bas.
                """)
                select_obs=st.selectbox("Choisissez une observation", range(0,len(feats)+1))
                st.write("RandomForestRegressor")
                st_shap(shap.force_plot(explainer_rfr.expected_value, shap_values_rfr[select_obs,:], feats.iloc[select_obs,:]))
                st.write("GradientBoostingRegressor")
                st_shap(shap.force_plot(explainer_gbr.expected_value, shap_values_gbr[select_obs,:], feats.iloc[select_obs,:]))



                
                




            





