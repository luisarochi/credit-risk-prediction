import pandas as pd

def load_data(path):
    """
    Carga un archivo CSV y devuelve un DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset cargado correctamente con shape {df.shape}")
        return df
    except FileNotFoundError:
        print(" Archivo no encontrado.")
        return None
