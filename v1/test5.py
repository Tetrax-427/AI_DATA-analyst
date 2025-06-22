"""
Take csv path as command line arg
Then allows user to ask question based on the CSV data.

Agent, that classify if output requires Image or not

Generates matplotlib code for plotting 
Insight functions, that identifies the underlying assumptions.
Quality Check that finds if there are issues in data or not.
"""

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from get_llm_response import get_response

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print("CSV loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def generate_plot(df, query):
    system_prompt = f"""
    Given a Dataframe, and a user query, your job is to return the most appropriate python code
    That can be used to plot the data/graph that is being asked by the query.
    Response only with the Code line.
    Use only the columns present in the Dataframe, and create a step by step plan for query.
    For example,
    query, plot the histogram for attendence with 10 bukcets in blue color
    STEP:
    1. Identify the column for attendence.
    2. create histogram
    3. set bin to 10
    4. set color to blue
    5. set x,y label and title
    6. Create the python code for the above steps.

    Don't add columns that are not required.
    Make sure to use proper columns names, values etc, so that code doesn't fail.
    """

    user_query = f""" DATAFRAME : {df} \n\n QUERY :: {query} """
    response = get_response(system_prompt, user_query)
    return response

def ask_question(df, question):
    # Convert a sample of the dataframe to string for token limits
    data = df.to_csv(index=False)
    
    system_prompt = f"""
    You are a data analyst. Answer the following question based on the data.
    Break down the query in multiple steps and follow them to give the proper answer.
    
    For example, if query asks for all the names in a department.
    STEPS will be like:
    1. iterate over all data points, if department name matches mark it yes.
    2. Collect all yes marked data points
    3. return the collected data points.

    Verify your answer, that it is correct with respect to the data.
    Make sure you don't miss any data point and output in correct format.
    Seprate your steps from actual final output.
    Data sample:
    {data}
    """

    user_query = question
    response = get_response(system_prompt, user_query)
    
    return response

def generate_insight(df, query):
    print("\n[Insight generation triggered based on query]")
    
    # Simple automatic insights from the data
    numeric_summary = df.describe().to_string()
    correlation = df.corr(numeric_only=True).to_string()

    system_prompt = f"""You are a data analyst. Extract insights from the following dataset.
        Data Frame :
                {df}
        Data Summary:
                {numeric_summary}
        Correlation Matrix:
                {correlation}

        And help the user to solve the problem, by providing the best posible answer to the query.
        Get proper calculations done, dont answer just based on intution.
        Create proper step by step plan and execute it.
        For example, 
        query, what is the relation of student gender and department.
        STEPS:
        1. Collect all student data in one group.
        2. Calculate num of each gender in each group.
        3. Calculate exact values, percentage, ratio and other statistical numbers that might be useful.
        4. Analyze the numbers, w.r.t query.
        5. Return the findings.
        Answer in a professional tone.

    """
    user_query = f"{query}" 

    insights = get_response(system_prompt, user_query)
    return insights

def check_data_quality(df, query):
    print("\n[Quality Check triggered based on query]")
    
    # Simple automatic insights from the data
    numeric_summary = df.describe().to_string()
    correlation = df.corr(numeric_only=True).to_string()

    system_prompt = f"""You are a data analyst. Extract insights from the following dataset.
        Data Frame :
                {df}
        Data Summary:
                {numeric_summary}
        Correlation Matrix:
                {correlation}

        And help the user to solve the problem, by providing the best possible answer to the query.
        Get proper calculations done, dont answer just based on intution.
        Create proper step by step plan and execute it.
        
        Identify issues in data provided using whole data or summary.
        Pin point the issues, i.e. the row and column.

        For example,
        query, is there any issue with GPA column.
        SETPS:
        1. Identify the column related to GPA.
        2. Check if some data is missing.
        3. Check if values are numeric only.
        4. Check if all values lies in a jusified range like 0-4,0-10 
        5. return the findings.
        Answer in a professional tone.

    """
    user_query = f"{query}" 

    quality_checks = get_response(system_prompt, user_query)
    return quality_checks

def update_data(df, query):
    print("\n[Data Updation triggered based on query]")
    
    system_prompt = f"""
        Help user with data updaton.
        Given a DataFrame, give user the python code to do the updation.
        Don't update the dataframe, just return the PYTHON code for updation, that creates new CSV File
        Create a step by step plan to complete the plan.

        For example, 
        query, Update all the GPAs to scale of 10.
        STEPS:
        1. Identify the col of GPA.
        2. Identify original scale of GPA.
        3. Create formula for updation.
        4. Apply formula for all rows.
        5. Update the Column

        Always add the code to save the DataFrame to CSV.
        And use "file_path" as variable name to read CSV.
        Return Python code.
    """
    user_query = f"DataFrame : {df}, \n\n QUERY : {query}" 

    updates = get_response(system_prompt, user_query)
    return updates

def classify_query(user_query):
    system_prompt = (
        "You are an intelligent agent. Your task is to classify the user's query into one of three categories:\n\n"
        "1. 'graph' - if the query is asking for a chart, visual, or plot.\n"
        "2. 'insight' - if the query asks for trends, patterns, correlations, or underlying information in the data.\n"
        "3. 'quality_check' - if the query asks for some quality checks related to data."
        "4. 'update_data' - if query requires the updation of original data."
        "5. 'text' - if the query just requires a basic textual answer from the data.\n\n"
        "Respond with ONLY one word: graph, insight, or text."
    )
    decision = get_response(system_prompt, user_query).strip().lower()
    return decision

import pandas as pd
from collections import Counter

import pandas as pd
import re
from collections import Counter

def initial_data_check(df: pd.DataFrame):
    print("\n🔍 Initial Dataset Check\n" + "-" * 40)
    
    print(f"\n🧾 Total Rows: {len(df)}")
    print(f"🧾 Total Columns: {len(df.columns)}\n")

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

    email_regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    mobile_regex = re.compile(r"^\+?[1-9]\d{9,14}$")  # E.164-like international format

    for col in df.columns:
        print(f"\n📌 Column: {col}")
        col_data = df[col]
        
        # Check for multiple Python types
        actual_types = col_data.dropna().map(type).value_counts()
        type_names = [t.__name__ for t in actual_types.index]
        multiple_types = len(type_names) > 1

        inferred_type = infer_dtype(col_data)
        type_flag = f"{inferred_type} ({'MIXED TYPES: ' + ', '.join(type_names)})" if multiple_types else inferred_type
        print(f"  • Inferred Type: {type_flag}")
        
        total_nulls = col_data.isnull().sum()
        total_not_nulls = col_data.notnull().sum()
        unique_vals = col_data.nunique(dropna=True)
        print(f"  • Null Values: {total_nulls}")
        print(f"  • Non-Null Values: {total_not_nulls}")
        print(f"  • Unique Values: {unique_vals}")

        # Special checks
        if inferred_type == "datetime":
            try:
                datetimes = pd.to_datetime(col_data, errors="coerce")
                invalid_dates = datetimes.isnull().sum()
                out_of_range = ((datetimes.dt.year < 1900) | (datetimes.dt.year > 2100)).sum()
                print(f"  • Invalid datetime values: {invalid_dates}")
                print(f"  • Out-of-range years: {out_of_range}")
            except Exception as e:
                print(f"  • ⚠️ Error parsing datetimes: {e}")

        elif inferred_type == "str":
            valid_emails = col_data.dropna().apply(lambda x: bool(email_regex.fullmatch(str(x)))).sum()
            valid_mobiles = col_data.dropna().apply(lambda x: bool(mobile_regex.fullmatch(str(x)))).sum()
            if valid_emails > 0:
                print(f"  • ✅ Email-like values: {valid_emails}")
            if valid_mobiles > 0:
                print(f"  • ✅ Mobile number-like values: {valid_mobiles}")

        # Numerical Stats
        if inferred_type in ["int", "float"]:
            print(f"  • Mean: {col_data.mean():.2f}")
            print(f"  • Median: {col_data.median():.2f}")
            print(f"  • Mode: {col_data.mode().values[:1]}")
            print(f"  • Std Dev: {col_data.std():.2f}")
            print(f"  • Min: {col_data.min()}")
            print(f"  • Max: {col_data.max()}")

        elif inferred_type in ["str", "bool"]:
            mode_vals = col_data.mode().values[:1]
            print(f"  • Mode: {mode_vals}")

        # Top 10 Frequent Values
        top_vals = col_data.value_counts(dropna=False).head(10)
        print("  • Top 10 Values:")
        for val, count in top_vals.items():
            print(f"     - {repr(val)}: {count} times")

    print("\n✅ Dataset profiling complete.\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python ai_analyst.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_csv(file_path)
    initial_data_check(df)
    while True:
        question = input("\nAsk a question about the data (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        decision = classify_query(question)
        if decision == "graph":
            answer = generate_plot(df, question)
        elif decision == "insight":
            answer = generate_insight(df, question)
        elif decision == "quality_check":
            answer = check_data_quality(df, question)
        elif decision == "update_data":
            answer = update_data(df,question)
        else:
            answer = ask_question(df, question)
        print("\nAI Analyst:", answer)

if __name__ == "__main__":
    main()

