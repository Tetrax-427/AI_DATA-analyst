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

def generate_column_summary(df):
    col_summaries = {}
    for col in df.columns:
        col_data = df[col]
        summary = {}
        summary['Type'] = infer_dtype(col_data)
        summary['Nulls'] = int(col_data.isnull().sum())
        summary['Non-Nulls'] = int(col_data.notnull().sum())
        summary['Unique'] = int(col_data.nunique(dropna=True))

        if summary['Type'] in ["int", "float"]:
            summary['Mean'] = float(col_data.mean())
            summary['Median'] = float(col_data.median())
            summary['Mode'] = col_data.mode().values[:1].tolist()
            summary['Std Dev'] = float(col_data.std())
            summary['Min'] = float(col_data.min())
            summary['Max'] = float(col_data.max())
        elif summary['Type'] == "str":
            summary['Mode'] = col_data.mode().values[:1].tolist()

        top_vals = col_data.value_counts(dropna=False).head(5).to_dict()
        summary['Top Values'] = top_vals

        col_summaries[col] = summary
    return col_summaries

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
        st.session_state.chat_history = []  # Clear chat history on new upload
        st.success("File uploaded successfully!")

    except Exception as e:
        st.error(f"Error reading file: {e}")

# ------------------------------
# Data Summary and Chat Section
# ------------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("### 🔍 Initial Data Summary")
    st.info(f"**Total Rows:** {len(df)}  |  **Total Columns:** {len(df.columns)}")
    
    summaries = generate_column_summary(df)
    for col, summary in summaries.items():
        with st.expander(f"📌 {col}  ({summary['Type']})"):
            st.table(pd.DataFrame.from_dict({k: v for k, v in summary.items() if k != 'Top Values'}, orient='index', columns=['Value']))
            st.markdown("**Top Values:**")
            st.table(pd.DataFrame.from_dict(summary['Top Values'], orient='index', columns=['Count']))

    # st.markdown("---")
    # st.subheader("💬 Chat with your data")
    

    # Display chat history ABOVE the input
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Chat History")
        for user_query, bot_reply in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**🧑 User:** {user_query}")
                st.markdown(f"**🤖 AI:** {bot_reply}")
                st.markdown("---")

    st.subheader("💬 Chat with your data")
    user_input = st.text_input("Ask a question about your data:", key="chat_input", value="")

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

    
