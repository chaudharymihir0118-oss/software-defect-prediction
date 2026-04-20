import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from preprocessing import preprocess_data, split_data
from model import train_models, save_model

st.set_page_config(page_title="Software Defect Prediction")

st.title("Software Defect Prediction System")

menu = ["Home", "Upload Dataset", "Train Model", "Predict"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------------- HOME ----------------
if choice == "Home":
    st.write("This system predicts software defects using Machine Learning.")

# ---------------- UPLOAD ----------------
elif choice == "Upload Dataset":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write(df.head())
        st.session_state["data"] = df
        st.success("Dataset uploaded!")

# ---------------- TRAIN ----------------
elif choice == "Train Model":
    if "data" in st.session_state:

        df = st.session_state["data"]

        X, y, scaler = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        results = train_models(X_train, X_test, y_train, y_test)

        for model_name, res in results.items():
            st.subheader(model_name)
            st.write("Accuracy:", res["accuracy"])
            st.text(res["report"])

            cm = res["confusion_matrix"]
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        best_model = results[best_model_name]["model"]

        st.success(f"Best Model: {best_model_name}")

        save_model(best_model)

        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    else:
        st.warning("Upload dataset first!")

# ---------------- PREDICT ----------------
elif choice == "Predict":
    st.subheader("Enter Feature Values")

    loc = st.number_input("Lines of Code", min_value=0.0)
    complexity = st.number_input("Complexity", min_value=0.0)
    coupling = st.number_input("Coupling", min_value=0.0)
    cohesion = st.number_input("Cohesion", min_value=0.0)

    if st.button("Predict"):
        try:
            model = pickle.load(open("saved_model.pkl", "rb"))
            scaler = pickle.load(open("scaler.pkl", "rb"))

            data = [[loc, complexity, coupling, cohesion]]
            data = scaler.transform(data)

            prediction = model.predict(data)

            if prediction[0] == 1:
                st.error("Defective Module ❌")
            else:
                st.success("Non-Defective Module ✅")

        except:
            st.error("Train model first!")