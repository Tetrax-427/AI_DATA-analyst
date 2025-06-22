import streamlit as st
import pandas as pd
from io import StringIO
from get_llm_response import get_response
from req_functions import classify_query, generate_plot, generate_insight, check_data_quality, update_data, ask_question

# ------------------------------
# Session State Initialization
# ------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama-3.3-70b-versatile"
if "temp" not in st.session_state:
    st.session_state.temp = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 300

# ------------------------------
# Helper Functions
# ------------------------------
def initial_data_check(df: pd.DataFrame) -> str:
    summary_lines = []

    def infer_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return "int"
        elif pd.api.types.is_float_dtype(series):
            return "float"
        elif pd.api.types.is_bool_dtype(series):
            return "bool"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_string_dtype(series):
            return "str"
        else:
            return "unknown"

    summary_lines.append(f"\nTotal Rows: {len(df)}")
    summary_lines.append(f"Total Columns: {len(df.columns)}\n")

    for col in df.columns:
        col_data = df[col]
        inferred_type = infer_dtype(col_data)
        total_nulls = col_data.isnull().sum()
        total_not_nulls = col_data.notnull().sum()
        unique_vals = col_data.nunique(dropna=True)

        summary_lines.append(f"\nColumn: {col}")
        summary_lines.append(f"  • Inferred Type: {inferred_type}")
        summary_lines.append(f"  • Null Values: {total_nulls}")
        summary_lines.append(f"  • Non-Null Values: {total_not_nulls}")
        summary_lines.append(f"  • Unique Values: {unique_vals}")

        if inferred_type in ["int", "float"]:
            summary_lines.append(f"  • Mean: {col_data.mean():.2f}")
            summary_lines.append(f"  • Median: {col_data.median():.2f}")
            summary_lines.append(f"  • Mode: {col_data.mode().values[:1]}")
            summary_lines.append(f"  • Std Dev: {col_data.std():.2f}")
            summary_lines.append(f"  • Min: {col_data.min()}")
            summary_lines.append(f"  • Max: {col_data.max()}")

        elif inferred_type == "str":
            mode_vals = col_data.mode().values[:1]
            summary_lines.append(f"  • Mode: {mode_vals}")

        top_vals = col_data.value_counts(dropna=False).head(5)
        summary_lines.append("  • Top 5 Values:")
        for val, count in top_vals.items():
            summary_lines.append(f"     - {repr(val)}: {count} times")

    return "\n".join(summary_lines)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("📊 AI Data Analyst")

with st.sidebar:
    st.title("🔧 Tools")
    st.session_state.model_name = st.selectbox("Select LLM Model",
                                               ["llama-3.3-70b-versatile",
                                                "llama-3.1-8b-instant",
                                                "qwen-qwq-32b",
                                                "qwen/qwen3-32b",
                                                "deepseek-r1-distill-llama-70b",
                                                "mistral-saba-24b"],
                                               index=0)
    with st.expander("LLM Options"):
        st.session_state.temp = st.slider("Temperature", 0.0, 1.0, st.session_state.temp, step=0.05)
        st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.top_p, step=0.05)
        st.session_state.max_tokens = st.number_input("Max New Tokens", min_value=50, max_value=2048, value=st.session_state.max_tokens)

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.session_state.chat_history = []  # Clear chat history when new file is uploaded

        st.success("File uploaded successfully!")

        with st.expander("🔍 Initial Data Summary", expanded=True):
            summary_text = initial_data_check(df)
            st.text(summary_text)

    except Exception as e:
        st.error(f"Error reading file: {e}")

# Chat Interface
if st.session_state.df is not None:
    st.subheader("💬 Chat with your data")
    user_input = st.text_input("Ask a question about your data:", key="chat_input")

    if st.button("Send") and user_input.strip() != "":
        decision = classify_query(user_input)
        df = st.session_state.df

        if decision == "graph":
            answer = generate_plot(df, user_input)
        elif decision == "insight":
            answer = generate_insight(df, user_input)
        elif decision == "quality_check":
            answer = check_data_quality(df, user_input)
        elif decision == "update_data":
            answer = update_data(df, user_input)
        else:
            answer = ask_question(df, user_input)

        st.session_state.chat_history.append((user_input, answer))

    for user_query, bot_reply in st.session_state.chat_history:
        st.markdown(f"**User:** {user_query}")
        st.markdown(f"**AI:** {bot_reply}")
