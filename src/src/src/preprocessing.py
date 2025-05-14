import pandas as pd

def clean_column_names(df):
    """
    Limpia los nombres de las columnas: minúsculas y sin espacios.
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

def handle_missing_values(df):
    """
    Rellena o elimina valores nulos según el contexto de cada columna.
    """
    # Rellenar ingresos con la mediana si hay valores nulos
    if 'monthly_inc' in df.columns:
        df['monthly_inc'].fillna(df['monthly_inc'].median(), inplace=True)

    # Rellenar dependents con 0 si hay nulos
    if 'dependents' in df.columns:
        df['dependents'].fillna(0, inplace=True)

    return df

def remove_outliers(df, columns, z_thresh=3):
    """
    Elimina outliers simples usando el criterio de Z-score.
    """
    from scipy.stats import zscore
    df = df[(zscore(df[columns]) < z_thresh).all(axis=1)]
    return df
