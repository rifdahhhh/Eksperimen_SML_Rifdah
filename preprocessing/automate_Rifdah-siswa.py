import pandas as pd
import re

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
    """
    Membaca file CSV dan melakukan preprocessing:
    - Membersihkan nama kolom
    - Mengubah label target menjadi numerik
    - Mengembalikan X (fitur) dan y (label)

    Args:
        filepath: path ke file CSV

    Returns:
        Tuple (X, y) siap latih
    """
    df = pd.read_csv(filepath)
    df = clean_column_names(df)

    # Ubah label target ke biner (misalnya: 'lung_cancer' kolom target, ubah ke 0/1 jika perlu)
    if 'lung_cancer' in df.columns:
        df['lung_cancer'] = df['lung_cancer'].map({'YES': 1, 'NO': 0})

    # Tentukan kolom target dan fitur
    target_column = 'lung_cancer'
    feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns]
    y = df[target_column]

    return X, y
