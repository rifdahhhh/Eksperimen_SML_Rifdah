import pandas as pd
import re
import sys
import os

def clean_column_names(df):
    cleaned_columns = []
    for col in df.columns:
        cleaned_col = col.strip()
        cleaned_col = re.sub(r'[\s\-]+', '_', cleaned_col)
        cleaned_col = re.sub(r'[^\w]', '', cleaned_col)
        cleaned_col = cleaned_col.lower()
        cleaned_columns.append(cleaned_col)
    df.columns = cleaned_columns
    return df

def drop_duplicates(df):
    return df.drop_duplicates()

def map_gender(df, col_name='gender'):
    if col_name in df.columns:
        df[col_name] = df[col_name].map({'M': 0, 'F': 1}).fillna(-1).astype(int)
    return df

def map_lung_cancer(df, col_name='lung_cancer'):
    if col_name in df.columns:
        df[col_name] = df[col_name].map({'YES': 1, 'NO': 0}).fillna(-1).astype(int)
    return df

def map_binary_columns(df, binary_columns):
    existing_cols = [col for col in binary_columns if col in df.columns]
    df[existing_cols] = df[existing_cols].replace({1: 0, 2: 1})
    return df

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = clean_column_names(df)
    df = drop_duplicates(df)
    df = map_gender(df)
    df = map_lung_cancer(df)

    binary_cols = ['smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
                   'chronic_disease', 'fatigue', 'allergy', 'wheezing',
                   'alcohol_consuming', 'coughing', 'shortness_of_breath',
                   'swallowing_difficulty', 'chest_pain']
    df = map_binary_columns(df, binary_cols)

    target_column = 'lung_cancer'
    feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns]
    y = df[target_column]

    return X, y, df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python automate_Rifdah-siswa.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    X, y, df_cleaned = preprocess_data(input_path)

    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Simpan dataframe hasil preprocessing lengkap ke CSV
    df_cleaned.to_csv(output_path, index=False)
    print(f"Preprocessing selesai, file disimpan di {output_path}")
