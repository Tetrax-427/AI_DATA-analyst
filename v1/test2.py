"""
Take csv path as command line arg
Then allows user to ask question based on the CSV data.

Agent, that classify if output requires Image or not
Plotting not supported yet.
"""
import pandas as pd
import sys
from get_llm_response import get_response

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print("CSV loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def ask_question(df, question):
    # Convert a sample of the dataframe to string for token limits
    sample_data = df.to_csv(index=False)

    
    system_prompt = f"""
    You are a data analyst. Answer the following question based on the data.

    Data sample:
    {sample_data}
    """

    user_query = question
    response = get_response(system_prompt, user_query)
    
    return response

def classify_query(user_query):
    system_prompt = (
        "You are a smart agent. Your task is to decide if the user query about a dataset "
        "requires a visual output like a graph (bar chart, line plot, etc.) or not.\n"
        "Respond ONLY with 'graph' or 'text'."
    )
    decision = get_response(system_prompt, user_query).strip().lower()
    return decision

def generate_plot(df, query):
    return "Apolozies, but we don't support graph generation yet...."

def main():
    if len(sys.argv) != 2:
        print("Usage: python ai_analyst.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_csv(file_path)

    while True:
        question = input("\nAsk a question about the data (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        decision = classify_query(question)
        if decision == "graph":
            answer = generate_plot(df, question)
            print("\nAI Analyst:", answer)
        else:
            answer = ask_question(df, question)
            print("\nAI Analyst:", answer)

if __name__ == "__main__":
    main()

