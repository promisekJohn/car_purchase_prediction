
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Car Purchase AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        return pickle.load(file)


try:

    model = load_model()

except FileNotFoundError:

    st.error(
        "❌ model.pkl was not found.\n\n"
        "Make sure model.pkl is in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(f"❌ Error loading model: {e}")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚗 Car Purchase AI")

    st.divider()

    st.subheader("About")

    st.write(
        "This application uses a machine learning "
        "classification model to predict whether "
        "a customer is likely to purchase a car."
    )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write("**Model:** Classification")
    st.write("**Task:** Car Purchase Prediction")
    st.write("**Technology:** Python")
    st.write("**Framework:** Streamlit")

    st.divider()

    st.info(
        "Enter the customer's information and "
        "click the prediction button."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🚗 Car Purchase Prediction")

st.write(
    "### AI-powered customer purchase prediction system"
)

st.divider()


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

st.subheader("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="🤖 Model",
        value="Logistic Regression"
    )


with col2:

    st.metric(
        label="📊 Task",
        value="Classification"
    )


with col3:

    st.metric(
        label="🎯 Target",
        value="Car Purchase"
    )


with col4:

    st.metric(
        label="🟢 Status",
        value="Ready"
    )


st.divider()


# ============================================================
# CUSTOMER ASSESSMENT
# ============================================================

st.subheader("👤 Customer Assessment")

st.write(
    "Enter the customer's financial and demographic "
    "information below."
)


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

with col1:

    st.write("### 👤 Personal Information")

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    Gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    CreditScore = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

with col2:

    st.write("### 💰 Financial Information")

    AnnualSalary = st.number_input(
        "Annual Salary (₦)",
        min_value=0,
        value=5_000_000,
        step=100_000
    )

    Savings = st.number_input(
        "Savings (₦)",
        min_value=0,
        value=2_000_000,
        step=100_000
    )

    DebtToIncomeRatio = st.number_input(
        "Debt To Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )


# ============================================================
# EMPLOYMENT & CAR PREFERENCE
# ============================================================

st.write("### 🚘 Employment & Car Preference")

col1, col2 = st.columns(2)


with col1:

    YearsEmployed = st.number_input(
        "Years Employed",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )


with col2:

    CarAgePreference = st.number_input(
        "Preferred Car Age",
        min_value=0,
        max_value=20,
        value=5,
        step=1
    )


st.divider()


# ============================================================
# GENDER ENCODING
# ============================================================

if Gender == "Female":

    Gender_encoded = 0

else:

    Gender_encoded = 1


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.subheader("🔮 Prediction Center")

predict_button = st.button(
    "🚀 Predict Car Purchase",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Age": [Age],

        "Gender": [Gender_encoded],

        "AnnualSalary": [AnnualSalary],

        "CreditScore": [CreditScore],

        "YearsEmployed": [YearsEmployed],

        "DebtToIncomeRatio": [DebtToIncomeRatio],

        "Savings": [Savings],

        "CarAgePreference": [CarAgePreference]

    })


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )

        st.stop()


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")


    if prediction == "Yes":

        st.success(
            "🚗 LIKELY TO PURCHASE\n\n"
            "The customer is predicted to purchase a car."
        )

    else:

        st.error(
            "❌ UNLIKELY TO PURCHASE\n\n"
            "The customer is predicted not to purchase a car."
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = model.classes_


            st.subheader("📈 Prediction Probability")


            # ------------------------------------------------
            # PROBABILITY METRICS
            # ------------------------------------------------

            probability_cols = st.columns(len(classes))


            for i, class_name in enumerate(classes):

                probability = (
                    probabilities[i] * 100
                )


                with probability_cols[i]:

                    st.metric(
                        label=f"Prediction: {class_name}",
                        value=f"{probability:.2f}%"
                    )

                    st.progress(
                        float(probabilities[i])
                    )


            # =================================================
            # VISUALIZATION SECTION
            # =================================================

            st.divider()

            st.subheader("📊 Prediction Visualizations")

            st.write(
                "The charts below show the probability "
                "of the customer's car purchase prediction."
            )


            # ------------------------------------------------
            # CREATE TWO COLUMNS FOR CHARTS
            # ------------------------------------------------

            chart_col1, chart_col2 = st.columns(2)


            # =================================================
            # PIE CHART
            # =================================================

            with chart_col1:

                st.write("### 🥧 Purchase Probability")

                fig, ax = plt.subplots(
                    figsize=(6, 5)
                )

                ax.pie(
                    probabilities,
                    labels=classes,
                    autopct="%1.1f%%",
                    startangle=90
                )

                ax.set_title(
                    "Car Purchase Probability"
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)


            # =================================================
            # BAR CHART
            # =================================================

            with chart_col2:

                st.write("### 📊 Purchase Probability")

                fig, ax = plt.subplots(
                    figsize=(6, 5)
                )

                bars = ax.bar(
                    classes,
                    probabilities * 100
                )

                ax.set_title(
                    "Car Purchase Prediction"
                )

                ax.set_xlabel(
                    "Prediction"
                )

                ax.set_ylabel(
                    "Probability (%)"
                )

                ax.set_ylim(
                    0,
                    100
                )


                # ------------------------------------------------
                # ADD VALUES ABOVE BARS
                # ------------------------------------------------

                for bar, probability in zip(
                    bars,
                    probabilities * 100
                ):

                    ax.text(
                        bar.get_x()
                        + bar.get_width() / 2,

                        bar.get_height() + 2,

                        f"{probability:.1f}%",

                        ha="center",

                        fontsize=10
                    )


                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)


        except Exception as e:

            st.warning(
                f"Probability or charts could not be calculated: {e}"
            )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.divider()

    st.subheader("👤 Customer Summary")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Age",
            Age
        )


    with col2:

        st.metric(
            "Gender",
            Gender
        )


    with col3:

        st.metric(
            "Credit Score",
            CreditScore
        )


    with col4:

        st.metric(
            "Years Employed",
            YearsEmployed
        )


    # ========================================================
    # FINANCIAL SUMMARY
    # ========================================================

    st.subheader("💰 Financial Summary")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Annual Salary",
            f"₦{AnnualSalary:,.0f}"
        )


    with col2:

        st.metric(
            "Savings",
            f"₦{Savings:,.0f}"
        )


    with col3:

        st.metric(
            "Debt-to-Income Ratio",
            f"{DebtToIncomeRatio:.2f}"
        )


    # ========================================================
    # CAR PREFERENCE
    # ========================================================

    st.subheader("🚘 Car Preference")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Preferred Car Age",
            f"{CarAgePreference} years"
        )


    with col2:

        if prediction == "Yes":

            st.metric(
                "Purchase Decision",
                "YES 🚗"
            )

        else:

            st.metric(
                "Purchase Decision",
                "NO ❌"
            )


    # ========================================================
    # MODEL INPUT DATA
    # ========================================================

    st.divider()

    with st.expander("🔍 View Model Input Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD PREDICTION
    # ========================================================

    st.subheader("📥 Prediction Report")


    report = pd.DataFrame({

        "Age": [Age],

        "Gender": [Gender],

        "Annual Salary": [AnnualSalary],

        "Credit Score": [CreditScore],

        "Years Employed": [YearsEmployed],

        "Debt To Income Ratio": [
            DebtToIncomeRatio
        ],

        "Savings": [Savings],

        "Car Age Preference": [
            CarAgePreference
        ],

        "Prediction": [prediction]

    })


    csv = report.to_csv(index=False)


    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="car_purchase_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚗 Car Purchase Prediction AI | "
    "Built with Python, Pandas, Scikit-learn, "
    "Matplotlib & Streamlit"
)

