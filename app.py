import yaml
import tiktoken
import streamlit as st
import pandas as pd

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="OpenAI API Pricing Calculator",
    page_icon="💡",
    initial_sidebar_state="expanded",
)

# Load model attributes from YAML file
with open("config.yaml", "r") as file:
    models_data = yaml.safe_load(file)

# Dropdown to select input method
input_method = st.sidebar.selectbox(
    "Select input method",
    ("Text Area", "File Upload", "Input Tokens")
)

user_input = ""
input_token_count = 0  # Default value for input tokens
output_token_count = 1000000  # Default value for output tokens

if input_method == "Text Area":
    # Textbox for user input
    default_text = "a" * 10  # Default text with 100,000 characters
    user_input = st.sidebar.text_area("Enter your text to calculate token:", key="input", height=200, value=default_text)
    input_token_count = len(tiktoken.get_encoding("cl100k_base").encode(user_input))
elif input_method == "File Upload":
    # Function to handle file upload
    def handle_file_upload():
        uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "txt", "json"])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                return " ".join(df.iloc[:, 0].astype(str).tolist())
            elif uploaded_file.type == "text/plain":
                return uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/json":
                return pd.read_json(uploaded_file).to_string()
        return None

    # Handle file upload and update user input if a file is uploaded
    uploaded_text = handle_file_upload()
    if uploaded_text:
        user_input = uploaded_text
        input_token_count = len(tiktoken.get_encoding("cl100k_base").encode(user_input))
elif input_method == "Input Tokens":
    # Number input for input tokens
    input_token_count = st.sidebar.number_input("Enter number of input tokens", min_value=0, step=1, value=100000)

# Number input for output tokens
output_token_count = st.sidebar.number_input("Number of output tokens", min_value=0, step=1, value=100000)

# Columns for buttons
col1, col2, *cols = st.columns(8)

# Calculate pricing
with st.spinner():
    results = []
    for model_type, model_classes in models_data.items():
        for model_class in model_classes:
            for model in model_class["models"]:
                if model["name"].startswith("text-embedding"):
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    try:
                        encoding = tiktoken.encoding_for_model(model["name"])
                    except KeyError:
                        encoding = tiktoken.get_encoding("cl100k_base")

                if input_method != "Input Tokens":
                    input_token_count = len(encoding.encode(user_input))
                
                per_token = model.get("per_token", 1)
                input_cost_per_token = model.get("input_cost", "NA")
                output_cost_per_token = float(model.get("output_cost", 0))

                if model["name"] == "gpt-4-1106-vision-preview":
                    total_cost = (
                        (input_token_count) * input_cost_per_token / per_token
                        + output_token_count * output_cost_per_token / per_token
                    )
                else:
                    total_cost = (
                        input_token_count * input_cost_per_token / per_token
                        + output_token_count * output_cost_per_token / per_token
                    )

                results.append({
                    "Model Class": model_class["name"],
                    "Model": model["name"],
                    "Number of Characters": len(user_input),
                    "Number of Tokens": input_token_count,
                    "Number of Output Tokens": output_token_count,
                    "Input Cost per Token": f"${input_cost_per_token}",
                    "Output Cost per Token": f"${output_cost_per_token}",
                    "Total Cost": f"${total_cost:.7f}"
                })

    result_df = pd.DataFrame(results)
    st.table(result_df)
