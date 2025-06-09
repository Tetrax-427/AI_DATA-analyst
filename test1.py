"""
Take csv path as command line arg
Then allows user to ask question based on the CSV data.

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
        answer = ask_question(df, question)
        print("\nAI Analyst:", answer)

if __name__ == "__main__":
    main()
