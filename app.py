import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ====================================
# PAGE SETTINGS
# ====================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ====================================
# TITLE
# ====================================

st.title("📈 Stock Price Prediction Using Machine Learning")

st.write(
    "Upload any stock dataset CSV "
    "(must contain Open, High, Low, Close, Volume columns)"
)

# ====================================
# FILE UPLOAD
# ====================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ====================================
# AFTER FILE UPLOAD
# ====================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # ----------------------------
    # DATASET PREVIEW
    # ----------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head()
    )

    # ----------------------------
    # SHAPE
    # ----------------------------

    st.subheader(
        "Dataset Shape"
    )

    rows, cols = data.shape

    st.write(
        f"Rows: {rows}"
    )

    st.write(
        f"Columns: {cols}"
    )

    # ----------------------------
    # COLUMN NAMES
    # ----------------------------

    st.subheader(
        "Column Names"
    )

    column_df = pd.DataFrame(

        data.columns,

        columns=["Columns"]

    )

    st.table(
        column_df
    )

    # ----------------------------
    # MISSING VALUES
    # ----------------------------

    st.subheader(
        "Missing Values"
    )

    st.write(
        data.isnull().sum()
    )

    data = data.dropna()

    required = [

        "Open",
        "High",
        "Low",
        "Close",
        "Volume"

    ]

    # ----------------------------
    # CHECK REQUIRED COLUMNS
    # ----------------------------

    if all(
        col in data.columns
        for col in required
    ):

        # ------------------------
        # PRICE GRAPH
        # ------------------------

        st.subheader(
            "Closing Price Trend"
        )

        fig1, ax1 = plt.subplots(
            figsize=(10,5)
        )

        ax1.plot(
            data["Close"]
        )

        ax1.set_xlabel(
            "Days"
        )

        ax1.set_ylabel(
            "Closing Price"
        )

        st.pyplot(fig1)

        # ------------------------
        # HEATMAP
        # ------------------------

        st.subheader(
            "Correlation Heatmap"
        )

        fig2, ax2 = plt.subplots(
            figsize=(8,6)
        )

        sns.heatmap(

            data.corr(
                numeric_only=True
            ),

            annot=True,

            ax=ax2

        )

        st.pyplot(fig2)

        # ------------------------
        # FEATURES
        # ------------------------

        X = data[
            [
                "Open",
                "High",
                "Low",
                "Volume"
            ]
        ]

        y = data[
            "Close"
        ]

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42

        )

        # ------------------------
        # MODEL SELECT
        # ------------------------

        model_option = st.selectbox(

            "Select ML Model",

            [

                "Linear Regression",

                "Random Forest"

            ]

        )

        if model_option == "Linear Regression":

            model = LinearRegression()

        else:

            model = RandomForestRegressor(

                random_state=42

            )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        # ------------------------
        # PERFORMANCE
        # ------------------------

        mse = mean_squared_error(
            y_test,
            predictions
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        st.subheader(
            "Model Performance"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(

            "MSE",

            round(mse,2)

        )

        c2.metric(

            "MAE",

            round(mae,2)

        )

        c3.metric(

            "R² Score",

            round(r2,4)

        )

        # ------------------------
        # RESULTS TABLE
        # ------------------------

        results = pd.DataFrame({

            "Actual Price":

            y_test,

            "Predicted Price":

            predictions

        })

        st.subheader(

            "Actual vs Predicted Prices"

        )

        st.dataframe(

            results.head(20)

        )

        # ------------------------
        # ACTUAL VS PREDICTED GRAPH
        # ------------------------

        st.subheader(

            "Actual vs Predicted Graph"

        )

        fig3, ax3 = plt.subplots(
            figsize=(8,6)
        )

        ax3.scatter(

            y_test,

            predictions

        )

        ax3.set_xlabel(
            "Actual Price"
        )

        ax3.set_ylabel(
            "Predicted Price"
        )

        st.pyplot(fig3)

        # ------------------------
        # MANUAL INPUT
        # ------------------------

        st.subheader(

            "Predict Closing Price"

        )

        open_price = st.number_input(

            "Enter Open Price"

        )

        high_price = st.number_input(

            "Enter High Price"

        )

        low_price = st.number_input(

            "Enter Low Price"

        )

        volume = st.number_input(

            "Enter Volume"

        )

        # ------------------------
        # PREDICT BUTTON
        # ------------------------

        if st.button(

            "Predict"

        ):

            input_data = np.array([

                [

                    open_price,

                    high_price,

                    low_price,

                    volume

                ]

            ])

            prediction = model.predict(
                input_data
            )

            st.success(

                f"Predicted Closing Price: "

                f"{prediction[0]:.2f}"

            )

    else:

        st.error(

            "Dataset must contain:\n"

            "Open, High, Low, Close, Volume"

        )

else:

    st.info(

        "Upload a stock CSV dataset to begin"

    )