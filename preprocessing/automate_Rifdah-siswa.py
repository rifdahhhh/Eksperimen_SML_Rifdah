import pandas as pd
import re
import sys
import os

def clean_column_names(df):
    """
    Membersihkan nama kolom:
    - Menghapus spasi dan tanda hubung, diganti dengan underscore
    - Mengubah huruf menjadi lowercase
    - Menghapus karakter non-alfanumerik selain underscore

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame dengan nama kolom yang telah dibersihkan
    """
    cleaned_columns = []
    for col in df.columns:
        cleaned_col = col.strip()
        cleaned_col = re.sub(r'[\s\-]+', '_', cleaned_col)
        cleaned_col = re.sub(r'[^\w]', '', cleaned_col)
        cleaned_col = cleaned_col.lower()
        cleaned_columns.append(cleaned_col)
    df.columns = cleaned_columns
    return df

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = clean_column_names(df)

    if 'lung_cancer' in df.columns:
        df['lung_cancer'] = df['lung_cancer'].map({'YES': 1, 'NO': 0})

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