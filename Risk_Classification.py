import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics import precision_score, recall_score, f1_score
genai.configure(api_key="AIzaSyDeuEA5v4nfoDz93JGswyNMry3-hzqAdxc")
model = genai.GenerativeModel('gemini-2.0-flash')
st.set_page_config(page_title="LLM Risk Classifier")
st.title("LLM-based Risk Event Classifier")
uploaded_file = st.file_uploader("Upload a CSV file with 'Event_Description' and 'Label' columns", type="csv")
examples = []
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "Event_Description" not in df.columns or "Label" not in df.columns:
            st.error("CSV must contain 'Event_Description' and 'Label' columns.")
        else:
            st.success("File uploaded successfully.")
            st.dataframe(df.head())
            examples = df.sample(n=min(5, len(df))).to_dict(orient='records')
            st.subheader("💬 Classify a New Risk Event")
            user_input = st.text_area("Enter a new risk event description")
            if user_input and st.button("Classify"):
                prompt = "You are a classification assistant.\n"
                prompt += "Classify the following event as either 0 (Cybersecurity) or 1 (Financial).\n"
                prompt += "\nHere are some examples:\n"
                for ex in examples:
                    prompt += f"- \"{ex['Event_Description']}\" → {ex['Label']}\n"
                prompt += f"\nNow classify:\n\"{user_input}\"\nOnly reply with 0 or 1."

                try:
                    response = model.generate_content(prompt)
                    result = response.text.strip()
                    if result.startswith("0"):
                        st.success(" Predicted: Cybersecurity Risk (0)")
                    elif result.startswith("1"):
                        st.success(" Predicted: Financial Risk (1)")
                    else:
                        st.warning(f" Unexpected response: {result}")
                except Exception as e:
                    st.error(f" Error from Gemini API: {str(e)}")

            st.subheader(" Evaluate Model on Entire Dataset")
            if st.button("Evaluate Model Performance"):
                with st.spinner("Classifying all events in batch..."):
                    ground_truth = df["Label"].astype(str).tolist()
                    descriptions = df["Event_Description"].tolist()

                    prompt = "You are a classification assistant.\n"
                    prompt += "Classify each event below as either 0 (Cybersecurity) or 1 (Financial).\n"
                    prompt += "Reply with one label per line in the same order.\n"
                    prompt += "Examples:\n"
                    for ex in examples:
                        prompt += f"- \"{ex['Event_Description']}\" → {ex['Label']}\n"

                    prompt += "\nNow classify the following:\n"
                    for desc in descriptions:
                        prompt += f"- \"{desc}\"\n"

                    try:
                        response = model.generate_content(prompt)
                        result_lines = response.text.strip().splitlines()
                        predictions = [line.strip()[0] for line in result_lines if line.strip().startswith(("0", "1"))]

                        if len(predictions) != len(ground_truth):
                            st.warning(" The number of predictions doesn't match the number of input rows.")
                        else:
                            precision = precision_score(ground_truth, predictions, pos_label="1")
                            recall = recall_score(ground_truth, predictions, pos_label="1")
                            f1 = f1_score(ground_truth, predictions, pos_label="1")

                            st.subheader(" Evaluation Metrics")
                            st.write(f"Precision: {precision:.2f}")
                            st.write(f"Recall: {recall:.2f}")
                            st.write(f"F1 Score: {f1:.2f}")

                    except Exception as e:
                        st.error(f" Gemini API Error: {e}")

    except Exception as e:
        st.error(f" Failed to process file: {e}")
else:
    st.info("Upload a labeled dataset to get started.")

