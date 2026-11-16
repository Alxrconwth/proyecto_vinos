"""
=============================================================================
PROYECTO: CLASIFICADOR Y PREDICTOR DE ORIGEN Y CALIDAD DE VINOS
Módulo: app.py
Descripción: Servidor Web Flask interactivo para la predicción de origen
             geográfico/varietal y evaluación de calidad enológica de vinos.
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Ruta del archivo del modelo serializado
RUTA_MODELO = 'modelo_vinos.joblib'

# =============================================================================
# CATÁLOGO DE CARACTERÍSTICAS QUÍMICAS (EN ESPAÑOL)
# Cada variable incluye metadatos, unidades y opciones sugeridas predeterminadas
# =============================================================================
METADATOS_VARIABLES = {
    'alcohol': {
        'nombre': 'Grado Alcohólico',
        'unidad': '% vol',
        'categoria': 'alcohol_acidez',
        'descripcion': 'Porcentaje volumétrico de etanol. Otorga cuerpo, estructura y sensación cálida en boca.',
        'min': 10.5,
        'max': 15.5,
        'step': 0.01,
        'defecto': 13.00,
        'opciones_sugeridas': [
            {'etiqueta': 'Ligero y fresco (11.80 % vol)', 'valor': 11.80},
            {'etiqueta': 'Equilibrado / Medio (12.80 % vol)', 'valor': 12.80},
            {'etiqueta': 'Estructurado / Noble (13.70 % vol)', 'valor': 13.70},
            {'etiqueta': 'Potente y cálido (14.40 % vol)', 'valor': 14.40}
        ]
    },
    'malic_acid': {
        'nombre': 'Ácido Málico',
        'unidad': 'g/L',
        'categoria': 'alcohol_acidez',
        'descripcion': 'Ácido orgánico de sabor punzante característico de la uva verde. Aporta frescura y vivacidad.',
        'min': 0.5,
        'max': 6.0,
        'step': 0.01,
        'defecto': 2.34,
        'opciones_sugeridas': [
            {'etiqueta': 'Baja acidez / Suave (1.30 g/L)', 'valor': 1.30},
            {'etiqueta': 'Acidez media / Equilibrada (2.10 g/L)', 'valor': 2.10},
            {'etiqueta': 'Acidez viva y marcada (3.30 g/L)', 'valor': 3.30},
            {'etiqueta': 'Muy punzante (4.20 g/L)', 'valor': 4.20}
        ]
    },
    'ash': {
        'nombre': 'Cenizas Totales',
        'unidad': 'g/L',
        'categoria': 'alcohol_acidez',
        'descripcion': 'Residuo mineral inorgánico tras la incineración. Refleja la riqueza del suelo del viñedo.',
        'min': 1.2,
        'max': 3.5,
        'step': 0.01,
        'defecto': 2.37,
        'opciones_sugeridas': [
            {'etiqueta': 'Baja mineralidad (1.80 g/L)', 'valor': 1.80},
            {'etiqueta': 'Mineralidad media (2.35 g/L)', 'valor': 2.35},
            {'etiqueta': 'Suelo rico en sales (2.75 g/L)', 'valor': 2.75}
        ]
    },
    'alcalinity_of_ash': {
        'nombre': 'Alcalinidad de Cenizas',
        'unidad': 'meq/L',
        'categoria': 'alcohol_acidez',
        'descripcion': 'Capacidad tampón frente a los ácidos. Se relaciona con la salinidad y suavidad en paladar.',
        'min': 10.0,
        'max': 30.0,
        'step': 0.1,
        'defecto': 19.5,
        'opciones_sugeridas': [
            {'etiqueta': 'Baja alcalinidad (14.0 meq/L)', 'valor': 14.0},
            {'etiqueta': 'Media / Estándar (19.5 meq/L)', 'valor': 19.5},
            {'etiqueta': 'Alta capacidad tampón (24.0 meq/L)', 'valor': 24.0}
        ]
    },
    'magnesium': {
        'nombre': 'Magnesio',
        'unidad': 'mg/L',
        'categoria': 'minerales_aminoacidos',
        'descripcion': 'Mineral esencial para la nutrición de levaduras durante la fermentación alcohólica.',
        'min': 65.0,
        'max': 170.0,
        'step': 1.0,
        'defecto': 100.0,
        'opciones_sugeridas': [
            {'etiqueta': 'Moderado (88 mg/L)', 'valor': 88.0},
            {'etiqueta': 'Nivel medio (100 mg/L)', 'valor': 100.0},
            {'etiqueta': 'Rico en magnesio (125 mg/L)', 'valor': 125.0}
        ]
    },
    'total_phenols': {
        'nombre': 'Fenoles Totales',
        'unidad': 'mg/L',
        'categoria': 'polifenoles',
        'descripcion': 'Concentración global de compuestos antioxidantes. Base del color, textura y envejecimiento.',
        'min': 0.8,
        'max': 4.2,
        'step': 0.01,
        'defecto': 2.30,
        'opciones_sugeridas': [
            {'etiqueta': 'Bajo en polifenoles (1.45 mg/L)', 'valor': 1.45},
            {'etiqueta': 'Concentración media (2.30 mg/L)', 'valor': 2.30},
            {'etiqueta': 'Alto poder antioxidante (3.10 mg/L)', 'valor': 3.10}
        ]
    },
    'flavanoids': {
        'nombre': 'Flavonoides',
        'unidad': 'mg/L',
        'categoria': 'polifenoles',
        'descripcion': 'Polifenoles responsables de la estructura tánica, astringencia noble y longevidad del vino.',
        'min': 0.3,
        'max': 5.2,
        'step': 0.01,
        'defecto': 2.03,
        'opciones_sugeridas': [
            {'etiqueta': 'Tanino suave / Ligero (0.80 mg/L)', 'valor': 0.80},
            {'etiqueta': 'Estructura media / Sedoso (2.10 mg/L)', 'valor': 2.10},
            {'etiqueta': 'Gran Reserva / Tánico y noble (3.20 mg/L)', 'valor': 3.20}
        ]
    },
    'nonflavanoid_phenols': {
        'nombre': 'Fenoles No Flavonoides',
        'unidad': 'mg/L',
        'categoria': 'polifenoles',
        'descripcion': 'Ácidos fenólicos menores derivados principalmente de la pulpa y de la crianza en madera.',
        'min': 0.1,
        'max': 0.7,
        'step': 0.01,
        'defecto': 0.36,
        'opciones_sugeridas': [
            {'etiqueta': 'Bajo (0.24 mg/L)', 'valor': 0.24},
            {'etiqueta': 'Medio (0.35 mg/L)', 'valor': 0.35},
            {'etiqueta': 'Elevado (0.50 mg/L)', 'valor': 0.50}
        ]
    },
    'proanthocyanins': {
        'nombre': 'Proantocianidinas',
        'unidad': 'mg/L',
        'categoria': 'polifenoles',
        'descripcion': 'Taninos condensados que protegen el vino contra la oxidación y estabilizan el color rubí.',
        'min': 0.4,
        'max': 3.8,
        'step': 0.01,
        'defecto': 1.59,
        'opciones_sugeridas': [
            {'etiqueta': 'Tanino condensado suave (0.90 mg/L)', 'valor': 0.90},
            {'etiqueta': 'Nivel medio (1.60 mg/L)', 'valor': 1.60},
            {'etiqueta': 'Alta protección oxidativa (2.40 mg/L)', 'valor': 2.40}
        ]
    },
    'color_intensity': {
        'nombre': 'Intensidad de Color',
        'unidad': 'Abs',
        'categoria': 'color_espectro',
        'descripcion': 'Densidad óptica de la muestra. Indica la profundidad visual, capa y concentración de antocianos.',
        'min': 1.0,
        'max': 13.5,
        'step': 0.01,
        'defecto': 5.06,
        'opciones_sugeridas': [
            {'etiqueta': 'Capa baja / Rubí claro (2.80 Abs)', 'valor': 2.80},
            {'etiqueta': 'Capa media / Cereza brillante (5.20 Abs)', 'valor': 5.20},
            {'etiqueta': 'Capa alta / Granate oscuro (7.50 Abs)', 'valor': 7.50}
        ]
    },
    'hue': {
        'nombre': 'Matiz / Tonalidad',
        'unidad': 'Relación',
        'categoria': 'color_espectro',
        'descripcion': 'Cociente de absorbancia 420nm/520nm. Revela la juventud (violeta/rojo) o madurez (teja/caoba).',
        'min': 0.4,
        'max': 1.8,
        'step': 0.01,
        'defecto': 0.96,
        'opciones_sugeridas': [
            {'etiqueta': 'Maduro / Reflejos teja (0.72)', 'valor': 0.72},
            {'etiqueta': 'Equilibrado / Rojo rubí vivo (1.05)', 'valor': 1.05},
            {'etiqueta': 'Muy joven / Reflejos violáceos (1.25)', 'valor': 1.25}
        ]
    },
    'od280/od315_of_diluted_wines': {
        'nombre': 'Absorbancia OD280/OD315',
        'unidad': 'Índice',
        'categoria': 'color_espectro',
        'descripcion': 'Pureza espectrofotométrica. Mide el balance entre flavonoides y proteínas en solución diluida.',
        'min': 1.2,
        'max': 4.1,
        'step': 0.01,
        'defecto': 2.61,
        'opciones_sugeridas': [
            {'etiqueta': 'Pureza moderada (1.65)', 'valor': 1.65},
            {'etiqueta': 'Buena estabilidad (2.75)', 'valor': 2.75},
            {'etiqueta': 'Máxima pureza proteica (3.50)', 'valor': 3.50}
        ]
    },
    'proline': {
        'nombre': 'Prolina',
        'unidad': 'mg/L',
        'categoria': 'minerales_aminoacidos',
        'descripcion': 'Aminoácido predominante en la uva madura. Revela el grado de madurez fisiológica en vendimia.',
        'min': 250.0,
        'max': 1750.0,
        'step': 1.0,
        'defecto': 750.0,
        'opciones_sugeridas': [
            {'etiqueta': 'Vendimia joven (480 mg/L)', 'valor': 480.0},
            {'etiqueta': 'Madurez fenólica óptima (750 mg/L)', 'valor': 750.0},
            {'etiqueta': 'Gran Reserva / Alta concentración (1150 mg/L)', 'valor': 1150.0}
        ]
    }
}

CATEGORIAS = [
    {
        'id': 'alcohol_acidez',
        'titulo': 'Grado Alcohólico y Ácidos',
        'icono': '🍷',
        'descripcion': 'Parámetros que determinan la calidez, frescura y acidez en boca.'
    },
    {
        'id': 'polifenoles',
        'titulo': 'Compuestos Fenólicos y Antioxidantes',
        'icono': '🍇',
        'descripcion': 'Estructura tánica, capacidad antioxidante y potencial de guarda prolongada.'
    },
    {
        'id': 'color_espectro',
        'titulo': 'Color y Espectrofotometría',
        'icono': '🎨',
        'descripcion': 'Densidad cromática, matiz evolutivo y pureza proteica óptica.'
    },
    {
        'id': 'minerales_aminoacidos',
        'titulo': 'Minerales y Aminoácidos',
        'icono': '🌿',
        'descripcion': 'Extracto seco natural, nutrición de levaduras y maduración en vendimia.'
    }
]

# Información descriptiva de cada clase/origen de vino
DETALLES_ORIGEN = {
    0: {
        'id': 0,
        'nombre_corto': 'Cultivar Barolo',
        'nombre_completo': 'Cultivar 1 (Tipo Barolo - Piamonte DOCG)',
        'cepa': 'Uva Nebbiolo',
        'region': 'Región de Piamonte (Langhe / Barolo), Italia',
        'perfil': 'Vino tinto de altísima estirpe y gran estructura. Sobresale por su elevado contenido de prolina y flavonoides nobles, color rubí granate y taninos firmes pero aterciopelados.',
        'maridaje': 'Carnes rojas asadas, trufa negra, caza mayor y quesos añejos curados.',
        'guarda': '10 a 20 años en condiciones óptimas de cava.',
        'icono': '👑',
        'color_acento': '#722F37'
    },
    1: {
        'id': 1,
        'nombre_corto': 'Cultivar Grignolino',
        'nombre_completo': 'Cultivar 2 (Tipo Grignolino - Monferrato DOC)',
        'cepa': 'Uva Grignolino',
        'region': 'Región de Asti y Monferrato (Piamonte), Italia',
        'perfil': 'Vino tinto joven, fresco y expresivo. Presenta menor intensidad alcohólica y de color, con un carácter aromático floral, taninos suaves y frescura equilibrada.',
        'maridaje': 'Pastas frescas con salsa pomodoro, embutidos artesanos, aves y pescados grasos.',
        'guarda': '2 a 5 años (excelente momento de consumo actual).',
        'icono': '🌸',
        'color_acento': '#A4384B'
    },
    2: {
        'id': 2,
        'nombre_corto': 'Cultivar Barbera',
        'nombre_completo': 'Cultivar 3 (Tipo Barbera d’Alba / d’Asti DOCG)',
        'cepa': 'Uva Barbera',
        'region': 'Colinas de Alba y Asti (Piamonte), Italia',
        'perfil': 'Vino vibrante de color violáceo profundo y marcada acidez málica. Gran concentración de color y carácter muy gastronómico, directo y enérgico.',
        'maridaje': 'Risotto al funghi, estofados especiados, carpaccio y quesos semicurados.',
        'guarda': '4 a 8 años, ganando redondez con crianza en roble.',
        'icono': '🍇',
        'color_acento': '#531B23'
    }
}

# Catálogo completo de muestras predefinidas para selección directa
CATALOGO_MUESTRAS = {
    'barolo_reserva': {
        'nombre': 'Barolo Gran Reserva (Lote C0-A: Alta Prolina y Taninos)',
        'grupo': 'Cultivar Barolo (Clase 0)',
        'icono': '👑',
        'clase_esperada': 0,
        'valores': {
            'alcohol': 14.23, 'malic_acid': 1.71, 'ash': 2.43, 'alcalinity_of_ash': 15.60,
            'magnesium': 127.0, 'total_phenols': 2.80, 'flavanoids': 3.06, 'nonflavanoid_phenols': 0.28,
            'proanthocyanins': 2.29, 'color_intensity': 5.64, 'hue': 1.04,
            'od280/od315_of_diluted_wines': 3.92, 'proline': 1065.0
        }
    },
    'barolo_clasico': {
        'nombre': 'Barolo Tradicional (Lote C0-B: Estructurado y Seco)',
        'grupo': 'Cultivar Barolo (Clase 0)',
        'icono': '👑',
        'clase_esperada': 0,
        'valores': {
            'alcohol': 13.16, 'malic_acid': 2.36, 'ash': 2.67, 'alcalinity_of_ash': 18.60,
            'magnesium': 101.0, 'total_phenols': 2.80, 'flavanoids': 3.24, 'nonflavanoid_phenols': 0.30,
            'proanthocyanins': 2.81, 'color_intensity': 5.68, 'hue': 1.03,
            'od280/od315_of_diluted_wines': 3.17, 'proline': 1185.0
        }
    },
    'grignolino_fresco': {
        'nombre': 'Grignolino Fresco y Ligero (Lote C1-A: 12.3° alcohol)',
        'grupo': 'Cultivar Grignolino (Clase 1)',
        'icono': '🌸',
        'clase_esperada': 1,
        'valores': {
            'alcohol': 12.37, 'malic_acid': 0.94, 'ash': 1.36, 'alcalinity_of_ash': 10.60,
            'magnesium': 88.0, 'total_phenols': 1.98, 'flavanoids': 0.57, 'nonflavanoid_phenols': 0.28,
            'proanthocyanins': 0.42, 'color_intensity': 1.95, 'hue': 1.05,
            'od280/od315_of_diluted_wines': 1.82, 'proline': 520.0
        }
    },
    'grignolino_aromatico': {
        'nombre': 'Grignolino Joven Floral (Lote C1-B: Equilibrado)',
        'grupo': 'Cultivar Grignolino (Clase 1)',
        'icono': '🌸',
        'clase_esperada': 1,
        'valores': {
            'alcohol': 12.28, 'malic_acid': 1.93, 'ash': 2.24, 'alcalinity_of_ash': 20.20,
            'magnesium': 95.0, 'total_phenols': 2.26, 'flavanoids': 2.08, 'nonflavanoid_phenols': 0.36,
            'proanthocyanins': 1.63, 'color_intensity': 3.09, 'hue': 1.06,
            'od280/od315_of_diluted_wines': 2.79, 'proline': 520.0
        }
    },
    'barbera_acida': {
        'nombre': 'Barbera Clásica (Lote C2-A: Alta Acidez Málica 3.33 g/L)',
        'grupo': 'Cultivar Barbera (Clase 2)',
        'icono': '🍇',
        'clase_esperada': 2,
        'valores': {
            'alcohol': 13.15, 'malic_acid': 3.33, 'ash': 2.44, 'alcalinity_of_ash': 21.40,
            'magnesium': 99.0, 'total_phenols': 1.68, 'flavanoids': 0.78, 'nonflavanoid_phenols': 0.45,
            'proanthocyanins': 1.15, 'color_intensity': 7.40, 'hue': 0.68,
            'od280/od315_of_diluted_wines': 1.68, 'proline': 630.0
        }
    },
    'barbera_oscura': {
        'nombre': 'Barbera de Capa Intensa (Lote C2-B: Color Profundo 7.4 Abs)',
        'grupo': 'Cultivar Barbera (Clase 2)',
        'icono': '🍇',
        'clase_esperada': 2,
        'valores': {
            'alcohol': 12.88, 'malic_acid': 2.99, 'ash': 2.40, 'alcalinity_of_ash': 20.00,
            'magnesium': 104.0, 'total_phenols': 1.30, 'flavanoids': 1.22, 'nonflavanoid_phenols': 0.24,
            'proanthocyanins': 0.83, 'color_intensity': 5.40, 'hue': 0.74,
            'od280/od315_of_diluted_wines': 1.42, 'proline': 530.0
        }
    },
    'promedio': {
        'nombre': 'Perfil Promedio General de Viñedos',
        'grupo': 'Muestra Estándar',
        'icono': '⚖️',
        'clase_esperada': None,
        'valores': {k: v['defecto'] for k, v in METADATOS_VARIABLES.items()}
    }
}


def obtener_o_entrenar_modelo():
    """
    Carga el modelo serializado desde el disco. Si no existe, entrena
    uno nuevo con RandomForest y lo persiste para ejecuciones posteriores.
    """
    if os.path.exists(RUTA_MODELO):
        paquete = joblib.load(RUTA_MODELO)
        return paquete['modelo'], paquete['columnas']

    # Entrenamiento automático de respaldo
    print("-> Inicializando modelo Random Forest por primera vez...")
    datos = load_wine(as_frame=True)
    df = datos.frame
    df.to_csv('wine_data.csv', index=False)

    X = df.drop(columns=['target'])
    y = df['target']

    modelo = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
    modelo.fit(X, y)

    joblib.dump({
        'modelo': modelo,
        'columnas': X.columns.tolist(),
        'version': '2.0.0'
    }, RUTA_MODELO)

    return modelo, X.columns.tolist()


modelo_rf, columnas_modelo = obtener_o_entrenar_modelo()


def evaluar_calidad_enologica(valores):
    """
    Calcula un Índice de Calidad Enológica (ICE) en escala de 0 a 100 puntos
    a partir del balance polifenólico, extracto seco (prolina), pureza espectral
    y armonía alcohol/acidez.
    """
    base = 50.0

    # 1. Flavonoides (Estructura tánica y longevidad, máx 16 pts)
    fav = min(max(valores.get('flavanoids', 2.0), 0.3), 4.5)
    pts_fav = (fav / 4.5) * 16.0

    # 2. Prolina (Extracto seco y madurez fenólica, máx 14 pts)
    pro = min(max(valores.get('proline', 700.0), 250.0), 1600.0)
    pts_pro = ((pro - 250.0) / 1350.0) * 14.0

    # 3. Pureza espectrométrica OD280/OD315 (máx 12 pts)
    od = min(max(valores.get('od280/od315_of_diluted_wines', 2.5), 1.2), 3.9)
    pts_od = ((od - 1.2) / 2.7) * 12.0

    # 4. Color y Matiz armónico (máx 8 pts)
    matiz = valores.get('hue', 1.0)
    if 0.8 <= matiz <= 1.25:
        pts_color = 8.0
    elif 0.6 <= matiz <= 1.4:
        pts_color = 5.0
    else:
        pts_color = 2.0

    # 5. Equilibrio alcohólico (máx 10 pts)
    alc = valores.get('alcohol', 13.0)
    if 12.5 <= alc <= 14.5:
        pts_alc = 10.0
    elif 11.5 <= alc < 12.5 or 14.5 < alc <= 15.0:
        pts_alc = 7.0
    else:
        pts_alc = 4.0

    # Penalización por acidez málica excesiva (> 3.8 g/L)
    malic = valores.get('malic_acid', 2.0)
    penalizacion_acidez = max(0.0, malic - 3.8) * 4.0

    puntaje_total = base + pts_fav + pts_pro + pts_od + pts_color + pts_alc - penalizacion_acidez
    puntaje_total = round(min(max(puntaje_total, 45.0), 98.5), 1)

    if puntaje_total >= 90.0:
        categoria = "Gran Reserva / Calidad Excepcional"
        medalla = "Oro"
        icono = "🏅"
        descripcion = "Excelente balance polifenólico, alta madurez y magnífico potencial de guarda."
    elif puntaje_total >= 80.0:
        categoria = "Reserva / Muy Buena Calidad"
        medalla = "Plata"
        icono = "🥈"
        descripcion = "Cuerpo equilibrado, taninos redondos y buena armonía organoléptica."
    elif puntaje_total >= 70.0:
        categoria = "Crianza / Calidad Buena y Equilibrada"
        medalla = "Bronce"
        icono = "🥉"
        descripcion = "Vino de perfil ágil, acidez agradable y taninos accesibles para consumo a medio plazo."
    else:
        categoria = "Vino Joven / Calidad de Mesa"
        medalla = "Estándar"
        icono = "🍷"
        descripcion = "Perfil ligero y fresco, pensado para consumo gastronómico inmediato."

    # Factores analizados para la ficha técnica
    factores = [
        {
            'factor': 'Potencial Tánico y Flavonoides',
            'valor': f"{valores.get('flavanoids', 0):.2f} mg/L",
            'estado': 'Sobresaliente' if fav >= 2.5 else ('Óptimo' if fav >= 1.5 else 'Moderado')
        },
        {
            'factor': 'Madurez Fenólica (Prolina)',
            'valor': f"{valores.get('proline', 0):.0f} mg/L",
            'estado': 'Alta Madurez' if pro >= 900 else ('Madurez Media' if pro >= 500 else 'Joven')
        },
        {
            'factor': 'Pureza Proteica (OD280/OD315)',
            'valor': f"{valores.get('od280/od315_of_diluted_wines', 0):.2f}",
            'estado': 'Excelente Estabilidad' if od >= 2.8 else ('Buena' if od >= 2.0 else 'Sensible')
        },
        {
            'factor': 'Grado Alcohólico',
            'valor': f"{valores.get('alcohol', 0):.1f} % vol",
            'estado': 'Cálido y Potente' if alc >= 13.5 else ('Equilibrado' if alc >= 12.0 else 'Ligero')
        }
    ]

    return {
        'puntaje': puntaje_total,
        'categoria': categoria,
        'medalla': medalla,
        'icono': icono,
        'descripcion': descripcion,
        'factores': factores
    }


def realizar_prediccion(valores_dict):
    """
    Ejecuta el pipeline completo de predicción de origen (Random Forest)
    y evaluación de calidad enológica para un vector de 13 características.
    """
    fila = [valores_dict[col] for col in columnas_modelo]
    df_entrada = pd.DataFrame([fila], columns=columnas_modelo)

    # 1. Predicción de Origen / Cultivar
    clase_predicha = int(modelo_rf.predict(df_entrada)[0])
    probabilidades_raw = modelo_rf.predict_proba(df_entrada)[0]

    probabilidades = []
    for idx, prob in enumerate(probabilidades_raw):
        info_cultivar = DETALLES_ORIGEN[idx]
        probabilidades.append({
            'clase': idx,
            'nombre': info_cultivar['nombre_corto'],
            'porcentaje': round(float(prob * 100), 1),
            'es_ganador': (idx == clase_predicha)
        })

    # Ordenar probabilidades de mayor a menor
    probabilidades.sort(key=lambda x: x['porcentaje'], reverse=True)

    # 2. Evaluación de Calidad Enológica
    calidad = evaluar_calidad_enologica(valores_dict)

    # 3. Datos del origen predicho
    info_origen = DETALLES_ORIGEN[clase_predicha]

    return {
        'origen': info_origen,
        'probabilidades': probabilidades,
        'calidad': calidad,
        'valores_ingresados': valores_dict
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    """Ruta principal: renderiza la interfaz web interactiva en español."""
    resultado = None
    error_mensaje = None
    muestra_seleccionada = request.form.get('selector_muestra_activa', '')
    datos_actuales = {k: v['defecto'] for k, v in METADATOS_VARIABLES.items()}

    if request.method == 'POST':
        try:
            valores_ingresados = {}
            for col in columnas_modelo:
                val_str = request.form.get(col, '')
                if val_str == '':
                    raise ValueError(f"El campo '{METADATOS_VARIABLES[col]['nombre']}' no puede estar vacío.")
                val_float = float(val_str)
                valores_ingresados[col] = val_float
                datos_actuales[col] = val_float

            resultado = realizar_prediccion(valores_ingresados)
        except Exception as e:
            error_mensaje = f"Error al procesar los datos: {str(e)}"

    return render_template(
        'index.html',
        categorias=CATEGORIAS,
        variables=METADATOS_VARIABLES,
        columnas=columnas_modelo,
        datos=datos_actuales,
        catalogo=CATALOGO_MUESTRAS,
        muestra_activa=muestra_seleccionada,
        resultado=resultado,
        error=error_mensaje
    )


@app.route('/api/predecir', methods=['POST'])
def api_predecir():
    """
    Endpoint API REST para predicciones en formato JSON.
    Permite integrar este clasificador con otros servicios o pruebas externas.
    """
    try:
        contenido = request.get_json(force=True)
        valores = {}
        for col in columnas_modelo:
            if col not in contenido:
                return jsonify({
                    'exito': False,
                    'error': f"Falta el parámetro requerido '{col}' ({METADATOS_VARIABLES[col]['nombre']})"
                }), 400
            valores[col] = float(contenido[col])

        prediccion = realizar_prediccion(valores)
        return jsonify({
            'exito': True,
            'origen_predicho': prediccion['origen']['nombre_completo'],
            'variedad_cepa': prediccion['origen']['cepa'],
            'region': prediccion['origen']['region'],
            'indice_calidad': prediccion['calidad']['puntaje'],
            'categoria_calidad': prediccion['calidad']['categoria'],
            'medalla': prediccion['calidad']['medalla'],
            'probabilidades': prediccion['probabilidades'],
            'factores_clave': prediccion['calidad']['factores']
        })
    except Exception as e:
        return jsonify({'exito': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("   SERVIDOR FLASK DE PREDICCIÓN DE VINOS ACTIVO")
    print("   Abre tu navegador en: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
