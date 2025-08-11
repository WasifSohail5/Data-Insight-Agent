import os
import pandas as pd
from langchain_core.tools import tool
import openpyxl
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def load_gemini_llm(model="gemini-1.5-flash"):
    return ChatGoogleGenerativeAI(model=model,temperature=0.3,convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"))

@tool
def preprocess_dataset(file_path:str)->str:
    """
        Automatically preprocesses the dataset at the given file path using Gemini 1.5 Flash.
        Returns a detailed explanation of the preprocessing steps.
        The cleaned CSV will be saved to 'output/cleaned_dataset.csv'.
        """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        return "Unsupported file format. Please upload a CSV or Excel file."

    data_as_str=df.head(100).to_csv(index=False)
    prompt= f"""
You are a data preprocessing expert. Given this dataset (in CSV format), do the following:

1. Identify any missing values, outliers, or anomalies.
2. Suggest and apply appropriate preprocessing steps (imputation, encoding, scaling, etc.).
3. Provide a cleaned version of the dataset.
4. Generate a short report explaining what you did and why.

Dataset (first 100 rows):

{data_as_str}
"""
    llm=load_gemini_llm()
    response=llm.invoke(prompt)
    output_path = 'output/cleaned_dataset.csv'
    df.to_csv(output_path, index=False)
    return f"✅ Preprocessing complete.\n\n📄 Report:\n{response}\n\n💾 Cleaned dataset saved to: {output_path}"


def get_file_path():
    print("📂 Upload your dataset (CSV or Excel):")
    file_path = input("Enter full file path: ").strip()

    if not os.path.exists(file_path):
        print("❌ File not found.")
        return None

    if not file_path.endswith(('.csv', '.xlsx')):
        print("❌ Only .csv or .xlsx files are supported.")
        return None

    return file_path


def main():
    print("🔷 Data Insight Agent: Gemini-Powered Preprocessing")

    file_path = get_file_path()
    if not file_path:
        return

    print("\n🚀 Preprocessing your dataset...\n")
    result = preprocess_dataset.invoke(file_path)

    print("\n🎯 Result:")
    print(result)


if __name__ == "__main__":
    main()
