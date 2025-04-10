import streamlit as st
import json
import pandas as pd
import plotly.express as px

def show():
    st.title("📊 Model Training History Comparison")

    # === Load JSON function ===
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    # === Load histories ===
    paths = {
        "Baseline": './evaluation/baseline/training_history.json',
        "Pretrained Freeze": './evaluation/pretrained_freeze/training_history.json',
        "Pretrained Unfreeze": './evaluation/pretrained_unfreeze/training_history.json'
    }

    histories = {model: load_json(path) for model, path in paths.items()}

    # === Convert histories to DataFrame ===
    def history_to_df(history, model_name):
        return pd.DataFrame({
            "Epoch": [entry['epoch'] for entry in history],
            "Training Loss": [entry['train_loss'] for entry in history],
            "Validation Loss": [entry['val_loss'] for entry in history],
            "Validation Accuracy": [entry['val_acc'] * 100 for entry in history],
            "Model": model_name
        })

    all_dfs = [history_to_df(history, model) for model, history in histories.items()]
    combined_df = pd.concat(all_dfs)

    # === Model Selection (Optional — add dynamic checkboxes!) ===
    selected_models = st.multiselect(
        "Select models to display:",
        options=list(paths.keys()),
        default=list(paths.keys())
    )

    if selected_models:
        filtered_df = combined_df[combined_df["Model"].isin(selected_models)]

        # === Training Loss ===
        st.subheader("Training Loss Over Epochs")
        fig_train = px.line(filtered_df, x='Epoch', y='Training Loss', color='Model', markers=True)
        st.plotly_chart(fig_train)

        # === Validation Loss ===
        st.subheader("Validation Loss Over Epochs")
        fig_val_loss = px.line(filtered_df, x='Epoch', y='Validation Loss', color='Model', markers=True)
        st.plotly_chart(fig_val_loss)

        # === Validation Accuracy ===
        st.subheader("Validation Accuracy Over Epochs (%)")
        fig_val_acc = px.line(filtered_df, x='Epoch', y='Validation Accuracy', color='Model', markers=True)
        st.plotly_chart(fig_val_acc)

        # === Summary Table ===
        st.subheader("📝 Summary Table")

        summary_data = {
            "Model Type": [],
            "Final Training Loss": [],
            "Final Validation Loss": [],
            "Final Validation Accuracy (%)": []
        }

        for model in selected_models:
            df = combined_df[combined_df["Model"] == model]
            summary_data["Model Type"].append(model)
            summary_data["Final Training Loss"].append(df["Training Loss"].iloc[-1])
            summary_data["Final Validation Loss"].append(df["Validation Loss"].iloc[-1])
            summary_data["Final Validation Accuracy (%)"].append(df["Validation Accuracy"].iloc[-1])

        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
    else:
        st.warning("Please select at least one model to display the charts and summary.")

