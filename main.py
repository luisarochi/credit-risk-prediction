# main.py

import pandas as pd
from src.utils import load_data
from src.preprocessing import clean_column_names, handle_missing_values, remove_outliers
from src.modeling import split_data, train_model, evaluate_model

def main():
    # Paso 1: Cargar el dataset
    data_path = 'Data/Credit_Risk_Benchmark_Dataset.csv'
    df = load_data(data_path)

    if df is None:
        return

    # Paso 2: Limpieza y Preprocesamiento
    print("Limpieza de los datos")
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = remove_outliers(df, columns=['rev_util', 'debt_ratio', 'monthly_inc'])

    # Paso 3: Dividir los datos en entrenamiento y prueba
    print("Dividiendo los datos en conjuntos de entrenamiento y prueba")
    X_train, X_test, y_train, y_test = split_data(df)

    # Paso 4: Entrenar el modelo
    print("Entrenando el modelo...")
    model = train_model(X_train, y_train)

    # Paso 5: Evaluación del modelo
    print(" Evaluando el modelo")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
