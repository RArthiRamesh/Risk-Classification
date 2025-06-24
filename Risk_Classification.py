import streamlit as st
import pandas as pd
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyBv5NtUU65WZl-Is6QPrfkjMtwZBGSYiY0")

model = genai.GenerativeModel('gemini-2.0-flash')

st.set_page_config(page_title="LLM Risk Classifier")
st.title("🤖 LLM-based Risk Event Classifier")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file with 'Event_Description' and 'Label' columns", type="csv")

examples = []

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "Event_Description" not in df.columns or "Label" not in df.columns:
            st.error("CSV must contain 'Event_Description' and 'Label' columns.")
        else:
            st.success("✅ File uploaded successfully.")
            st.dataframe(df.head())

            # Build few-shot examples
            examples = df.sample(n=min(5, len(df))).to_dict(orient='records')

            # Input box
            st.subheader("💬 Classify a New Risk Event")
            user_input = st.text_area("Enter a new risk event description")

            if user_input and st.button("Classify"):
                # Construct prompt
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
                        st.success("🔐 Predicted: Cybersecurity Risk (0)")
                    elif result.startswith("1"):
                        st.success("💰 Predicted: Financial Risk (1)")
                    else:
                        st.warning(f"⚠️ Unexpected response: {result}")

                except Exception as e:
                    st.error(f"❌ Error from Gemini API: {str(e)}")
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("📤 Upload a labeled dataset to get started.")
