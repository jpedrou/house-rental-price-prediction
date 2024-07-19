import streamlit as st
import joblib as jb
import pandas as pd
import requests

 
def get_usd_to_brl_rate():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url)
        data = response.json()
        return data["rates"]["BRL"]
    except Exception:
        return 5


# ===============================================================
# Model Load
# ===============================================================

model = jb.load("./model/model.pkl")

# ===============================================================
# Streamlit Interface
# ===============================================================

st.set_page_config(layout="centered")

# Setting styles
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.image(image="../reports/giphy.gif", width=100)
st.sidebar.subheader("Choose an option to make a prediction.")
st.title("House Rental Price Prediction App")
st.markdown(
    "This app was built to make predictions of house rental prices. For more details, please access the project repository by clicking [here](https://github.com/jpedrou). "
)


city = st.sidebar.selectbox(
    label="Select a city",
    options=(
        "São Paulo",
        "Porto Alegre",
        "Rio de Janeiro",
        "Campinas",
        "Belo Horizonte",
    ),
)

st.sidebar.markdown("")

state = st.sidebar.selectbox(
    label="Select a state",
    options=(
        "SP",
        "RJ",
        "MG",
        "RS",
    ),
)

st.sidebar.markdown("")

furniture = st.sidebar.selectbox(
    label="Is it furnished",
    options=(
        "Yes",
        "No",
    ),
)

animals = st.sidebar.selectbox(
    label="Does it accept animals ?",
    options=(
        "Yes",
        "No",
    ),
)

st.sidebar.markdown("")

area = st.sidebar.number_input("Area")

st.sidebar.markdown("")

rooms = st.sidebar.number_input("Rooms Number", step=1)

st.sidebar.markdown("")

bathrooms = st.sidebar.number_input("Bathrooms Number", step=1)

st.sidebar.markdown("")

floors = st.sidebar.number_input("Floors Number", step=1)

st.sidebar.markdown("")

condominium = st.sidebar.number_input("Condominium Value")

st.sidebar.markdown("")

iptu = st.sidebar.number_input("IPTU Value")

st.sidebar.markdown("")

insurance = st.sidebar.number_input("Insurance Value")

st.sidebar.markdown("")

predict_btn = st.sidebar.button("Predict")

# ===============================================================
# Conditions
# ===============================================================

furniture = 1 if furniture == "Yes" else 0
animals = 1 if animals == "Yes" else 0

match city:
    case "São Paulo":
        city = 0

    case "Porto Alegre":
        city = 3

    case "Rio de Janeiro":
        city = 1

    case "Campinas":
        city = 4

    case "Belo Horizonte":
        city = 2

match state:
    case "MG":
        state = 2
    case "RJ":
        state = 1
    case "RS":
        state = 3
    case "SP":
        state = 0


# ===============================================================
# Predictions
# ===============================================================

if not predict_btn:
    st.title("No Predictions Available")
    st.subheader("Please, click on the button Predict to see the results!")

else:
    pred_df = pd.DataFrame()

    pred_df["cidade"] = [city]
    pred_df["estado"] = [state]
    pred_df["area"] = [area]
    pred_df["num_quartos"] = [rooms]
    pred_df["num_banheiros"] = [bathrooms]
    pred_df["num_andares"] = [floors]
    pred_df["aceita_animais"] = [animals]
    pred_df["mobilia"] = [furniture]
    pred_df["valor_condominio"] = [condominium]
    pred_df["valor_iptu"] = [iptu]
    pred_df["valor_seguro_incendio"] = [insurance]

    result = model.predict(pred_df)
    rate = get_usd_to_brl_rate()

    dolar_result = f"US$ {round(result[0] / rate, 2)}"
    real_result = f"R$ {round(result[0], 2)}"

    col0, col1 = st.columns(2, gap="small")
    st.markdown("")

    col0.metric(label="Rental Value Prediction (US$)", value=dolar_result)
    col1.metric(label="Rental Value Prediction (R$)", value=real_result)
