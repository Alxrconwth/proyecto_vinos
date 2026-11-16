"""
=============================================================================
PROYECTO: CLASIFICADOR Y PREDICTOR DE ORIGEN Y CALIDAD DE VINOS
Módulo: modelo_vinos.py
Descripción: Script de entrenamiento, evaluación exhaustiva y serialización
             del modelo de Machine Learning (Random Forest) para predecir
             el origen (cultivar) y calidad química de muestras de vino.
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Diccionario de traducción y nombres de características en español
NOMBRES_CARACTERISTICAS_ES = {
    'alcohol': 'Grado Alcohólico (% vol)',
    'malic_acid': 'Ácido Málico (g/L)',
    'ash': 'Cenizas (g/L)',
    'alcalinity_of_ash': 'Alcalinidad de Cenizas (meq/L)',
    'magnesium': 'Magnesio (mg/L)',
    'total_phenols': 'Fenoles Totales (mg/L)',
    'flavanoids': 'Flavonoides (mg/L)',
    'nonflavanoid_phenols': 'Fenoles No Flavonoides (mg/L)',
    'proanthocyanins': 'Proantocianidinas (mg/L)',
    'color_intensity': 'Intensidad de Color',
    'hue': 'Tonalidad / Matiz',
    'od280/od315_of_diluted_wines': 'Absorbancia OD280/OD315 (Pureza)',
    'proline': 'Prolina (mg/L)'
}

NOMBRES_CULTIVARES = {
    0: "Cultivar 1 (Tipo Barolo - Alta Estructura)",
    1: "Cultivar 2 (Tipo Grignolino - Cuerpo Medio / Fresco)",
    2: "Cultivar 3 (Tipo Barbera - Acidez Viva / Color Intenso)"
}


def cargar_y_exportar_datos():
    """
    Carga el dataset de vinos desde scikit-learn y genera los archivos CSV
    requeridos para la entrega académica (tanto en formato estándar como en español).
    """
    print("\n[1/5] Cargando conjunto de datos 'Wine Recognition'...")
    datos_vino = load_wine(as_frame=True)
    df = datos_vino.frame

    # 1. Guardar CSV estándar original (Requisito de la tarea)
    df.to_csv('wine_data.csv', index=False)
    print("   -> Generado con éxito: 'wine_data.csv' (formato estándar)")

    # 2. Guardar versión traducida al español con nombres descriptivos
    df_es = df.rename(columns=NOMBRES_CARACTERISTICAS_ES)
    df_es['cultivar_origen'] = df['target'].map(NOMBRES_CULTIVARES)
    df_es.to_csv('vinos_datos_espanol.csv', index=False, encoding='utf-8-sig')
    print("   -> Generado con éxito: 'vinos_datos_espanol.csv' (versión en español)")

    return df, datos_vino


def entrenar_y_evaluar_modelo(df):
    """
    Entrena el clasificador Random Forest y ejecuta una evaluación completa:
    Train/Test Split, Matriz de Confusión, Reporte de Clasificación,
    Validación Cruzada Estratificada y Análisis de Importancia de Variables.
    """
    X = df.drop(columns=['target'])
    y = df['target']

    print(f"\n[2/5] Preparando partición de datos (80% entrenamiento, 20% prueba)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   -> Muestras de entrenamiento: {len(X_train)}")
    print(f"   -> Muestras de prueba:        {len(X_test)}")

    print("\n[3/5] Entrenando modelo Random Forest Classifier (100 árboles)...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight='balanced'
    )
    modelo.fit(X_train, y_train)

    print("\n[4/5] Evaluando desempeño del modelo...")
    y_pred = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    # Validación cruzada K-Fold (5 pliegues)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_cv = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy') * 100

    # Imprimir informe formateado
    print("\n" + "=" * 65)
    print("          REPORTE DE EVALUACIÓN DE MACHINE LEARNING")
    print("=" * 65)
    print(f" Exactitud en Datos de Prueba (Accuracy): {exactitud:.2f}%")
    print(f" Puntuación F1 Ponderada (F1-Score):     {f1:.2f}%")
    print(f" Validación Cruzada (5 Pliegues):         {scores_cv.mean():.2f}% (+/- {scores_cv.std():.2f}%)")
    print("-" * 65)

    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    clases_cortas = ["Barolo (C0)", "Grignolino (C1)", "Barbera (C2)"]
    df_cm = pd.DataFrame(cm, index=[f"Real {c}" for c in clases_cortas],
                             columns=[f"Pred {c}" for c in clases_cortas])
    print(df_cm.to_string())

    print("\nReporte Detallado por Tipo de Vino:")
    reporte = classification_report(
        y_test, y_pred,
        target_names=[NOMBRES_CULTIVARES[i] for i in range(3)]
    )
    print(reporte)

    print("-" * 65)
    print("Importancia de las Características Químicas en la Predicción:")
    importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    for col, imp in importancias.items():
        nombre_amigable = NOMBRES_CARACTERISTICAS_ES.get(col, col)
        barra = "█" * int(imp * 40)
        print(f" • {nombre_amigable:<36} {imp*100:>5.2f}%  {barra}")

    # Entrenar modelo final sobre todos los datos para la aplicación en producción
    modelo_final = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight='balanced'
    )
    modelo_final.fit(X, y)

    return modelo_final, X.columns.tolist()


def guardar_modelo(modelo, columnas):
    """
    Serializa y guarda el modelo entrenado junto con sus metadatos
    en un archivo binario joblib para ser consumido por Flask.
    """
    print("\n[5/5] Guardando modelo entrenado en disco...")
    paquete = {
        'modelo': modelo,
        'columnas': columnas,
        'nombres_es': NOMBRES_CARACTERISTICAS_ES,
        'nombres_clases': NOMBRES_CULTIVARES,
        'version': '2.0.0'
    }
    ruta_modelo = 'modelo_vinos.joblib'
    joblib.dump(paquete, ruta_modelo)
    print(f"   -> Modelo guardado exitosamente en '{ruta_modelo}'")


def main():
    print("=" * 65)
    print("   SISTEMA PREDICTIVO DE ORIGEN Y CALIDAD DE VINOS")
    print("        Entrenamiento y Evaluación de Modelos")
    print("=" * 65)

    df, _ = cargar_y_exportar_datos()
    modelo, columnas = entrenar_y_evaluar_modelo(df)
    guardar_modelo(modelo, columnas)

    print("\n" + "=" * 65)
    print("¡PROCESO COMPLETADO EXITOSAMENTE!")
    print("El modelo está listo para usarse con la aplicación web: python app.py")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
