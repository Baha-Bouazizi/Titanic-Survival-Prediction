import streamlit as st
import pickle
import pandas as pd
import altair as alt

# ----- Configuration page -----
st.set_page_config(
    page_title="🛳️ Titanic Survival Prediction",
    page_icon="🛳️",
    layout="centered",
)

# ----- Custom CSS -----
st.markdown(
    """
    <style>
        /* Background gradient */
        .stApp {
            /* Background gradient */
            background: linear-gradient(135deg, #1f4080 0%, #162850 100%);
            background-attachment: fixed;
            color: #ffffff;
        }
        /* Sidebar styling */
        section[data-testid="stSidebar"] > div:first-child {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(8px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        /* Inputs */
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div>input {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            color: #ffffff;
        }
        /* Tables inside expanders */
        .stApp table, .stApp th, .stApp td {
            color: #ffffff !important;
        }
        /* Pseudo-element to place the image */

    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Load model -----
try:
    with open("best_model.pkl3", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Le fichier 'best_model.pkl3' est introuvable. Vérifie son emplacement.")
    st.stop()

st.title("🛳️ Prédiction de survie - Titanic")
st.write(
    "Renseigne les informations du passager dans la barre latérale puis clique sur **Prédire** "
    "pour connaître ses chances de survie."
)

# ----- Sidebar Inputs -----
st.sidebar.header("Paramètres du passager")
pclass = st.sidebar.selectbox("Classe du passager (pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("Sexe", ["male", "female"])
age = st.sidebar.slider("Âge", 0, 100, 30)
sibsp = st.sidebar.number_input("Frères/soeurs & conjoints (sibsp)", 0, 10, 0)
parch = st.sidebar.number_input("Parents & enfants (parch)", 0, 10, 0)
fare = st.sidebar.number_input("Tarif payé (fare)", 0.0, 600.0, 50.0)
class_col = st.sidebar.selectbox("Classe (class)", ["First", "Second", "Third"])
who = st.sidebar.selectbox("Type de personne (who)", ["man", "woman", "child"])
adult_male = st.sidebar.selectbox("Adulte mâle ?", [True, False])
alone = st.sidebar.selectbox("Voyage seul ?", [True, False])

# ----- Prepare input dataframe -----
input_data = pd.DataFrame(
    {
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "class": [class_col],
        "who": [who],
        "adult_male": [adult_male],
        "alone": [alone],
    }
)

# ----- Prediction -----
if st.button("🔮 Prédire"):
    with st.spinner("Calcul en cours..."):
        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]

            # Probability chart
            gauge_df = pd.DataFrame(
                {"Probabilité": [proba * 100], "Catégorie": ["Survivre"]}
            )
            chart = (
                alt.Chart(gauge_df)
                .mark_bar(size=80)
                .encode(
                    x=alt.X("Catégorie:N", axis=None),
                    y=alt.Y("Probabilité:Q", scale=alt.Scale(domain=[0, 100])),
                    color=alt.condition(
                        alt.datum.Probabilité > 50,
                        alt.value("#2ecc71"),
                        alt.value("#e74c3c"),
                    ),
                    tooltip=["Probabilité"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

            # Textual feedback
            if prediction == 1:
                st.success(
                    f"✅ Le passager a survécu (probabilité : {proba:.2%})"
                )
            else:
                st.error(
                    f"❌ Le passager n'a pas survécu (probabilité de survie : {proba:.2%})"
                )
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")

    # Show input summary
    with st.expander("Voir les paramètres saisis"):
        st.table(input_data)
