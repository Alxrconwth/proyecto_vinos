# Sistema Inteligente de Clasificación y Calidad de Vinos

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Idioma](https://img.shields.io/badge/Idioma-Espa%C3%B1ol%20100%25-green.svg)](#)

> **Proyecto Académico de Machine Learning**  
> Aplicación web integral y script analítico para la predicción de **origen varietal (cultivar)** y estimación de **calidad enológica** de vinos a partir de 13 descriptores químicos espectrofotométricos.

---

## Tabla de Contenidos
1. [Descripción General](#-descripción-general)
2. [Estructura del Proyecto](#-estructura-del-proyecto)
3. [Conjunto de Datos (Dataset)](#-conjunto-de-datos-dataset)
4. [Modelo de Machine Learning](#-modelo-de-machine-learning)
5. [Evaluación de Calidad Enológica](#-evaluación-de-calidad-enológica)
6. [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
7. [Uso de la Aplicación Web](#-uso-de-la-aplicación-web)
8. [API REST (Endpoint JSON)](#-api-rest-endpoint-json)
9. [Glosario de Variables Químicas](#-glosario-de-variables-químicas)

---

## Descripción General

Este proyecto fue desarrollado como entrega académica con el objetivo de aplicar técnicas avanzadas de **Machine Learning Supervisado** a la enología computacional. 

El sistema resuelve dos necesidades fundamentales en la industria vitivinícola:
1. **Predicción de Origen / Cultivar**: Identifica con un **100% de exactitud** a cuál de las 3 regiones/cultivos italianos de Piamonte (*Barolo / Uva Nebbiolo*, *Grignolino* o *Barbera*) pertenece una muestra analizada.
2. **Evaluación de Calidad Enológica**: Determina un **Índice de Calidad Enológica (0 a 100 puntos)** y otorga una medalla (*Gran Reserva*, *Reserva*, *Crianza* o *Vino de Mesa*) a partir del balance polifenólico, extracto seco de prolina, pureza óptica proteica y equilibrio alcohólico.

---

## Estructura del Proyecto

```text
Proyecto_vinos/
│
├── app.py                      # Servidor Web Flask con endpoints y lógica de predicción
├── modelo_vinos.py             # Script de entrenamiento, evaluación completa y serialización
├── modelo_vinos.joblib         # Modelo Random Forest serializado y calibrado
├── wine_data.csv               # Dataset en formato CSV estándar (requisito de entrega)
├── vinos_datos_espanol.csv     # Dataset en español con columnas traducidas y nombres de cepas
├── requirements.txt            # Lista de dependencias del proyecto
├── README.md                   # Documentación principal organizada y en español
├── DOCUMENTACION.md            # Informe técnico-académico detallado para evaluación
│
└── templates/
    └── index.html              # Interfaz web sommelier estética, moderna y responsiva
```

---

## Conjunto de Datos (Dataset)

El proyecto utiliza el reconocido **Wine Recognition Dataset** (*Cortez et al. / scikit-learn*), compuesto por **178 muestras** de vinos elaborados en una misma región de Italia a partir de tres cultivares diferentes:

| Cultivar | Nombre / Cepa Representativa | Región Enológica | N° Muestras |
| :---: | :--- | :--- | :---: |
| **Clase 0** | Cultivar Barolo (Uva Nebbiolo) | Langhe / Piamonte DOCG | 59 |
| **Clase 1** | Cultivar Grignolino | Asti y Monferrato DOC | 71 |
| **Clase 2** | Cultivar Barbera | Barbera d'Alba / d'Asti DOCG | 48 |

### Las 13 Variables Físico-Químicas:

| N° | Variable Original | Nombre en Español | Unidad | Rango Típico |
|:--:|:---|:---|:---:|:---:|
| 1 | `alcohol` | Grado Alcohólico | % vol | 11.0 – 14.8 |
| 2 | `malic_acid` | Ácido Málico | g/L | 0.74 – 5.80 |
| 3 | `ash` | Cenizas Totales | g/L | 1.36 – 3.23 |
| 4 | `alcalinity_of_ash` | Alcalinidad de Cenizas | meq/L | 10.6 – 30.0 |
| 5 | `magnesium` | Magnesio | mg/L | 70.0 – 162.0 |
| 6 | `total_phenols` | Fenoles Totales | mg/L | 0.98 – 3.88 |
| 7 | `flavanoids` | Flavonoides | mg/L | 0.34 – 5.08 |
| 8 | `nonflavanoid_phenols` | Fenoles No Flavonoides | mg/L | 0.13 – 0.66 |
| 9 | `proanthocyanins` | Proantocianidinas | mg/L | 0.41 – 3.58 |
| 10 | `color_intensity` | Intensidad de Color | Abs | 1.28 – 13.0 |
| 11 | `hue` | Tonalidad / Matiz | Cociente | 0.48 – 1.71 |
| 12 | `od280/od315_of_diluted_wines` | Absorbancia OD280/OD315 | Índice | 1.27 – 4.00 |
| 13 | `proline` | Prolina | mg/L | 278 – 1680 |

---

## Modelo de Machine Learning

Se seleccionó el algoritmo **Random Forest Classifier** (Ensamble de 100 Árboles de Decisión con `class_weight='balanced'`) por las siguientes razones:
- **No linealidad:** Captura interacciones complejas entre compuestos fenólicos y acidez.
- **Robustez ante sobreajuste (Overfitting):** El remuestreo bootstrap reduce la varianza.
- **Interpretabilidad:** Permite extraer el índice de importancia de cada parámetro químico.

### Resultados de Evaluación (`modelo_vinos.py`):

```text
=================================================================
          REPORTE DE EVALUACIÓN DE MACHINE LEARNING
=================================================================
 Exactitud en Datos de Prueba (Accuracy): 100.00%
 Puntuación F1 Ponderada (F1-Score):     100.00%
 Validación Cruzada (5 Pliegues):         98.87% (+/- 1.38%)
-----------------------------------------------------------------

Matriz de Confusión:
                      Pred Barolo (C0)  Pred Grignolino (C1)  Pred Barbera (C2)
Real Barolo (C0)                    12                     0                  0
Real Grignolino (C1)                 0                    14                  0
Real Barbera (C2)                    0                     0                 10
```

### Importancia de las Características:
1. **Flavonoides** (`19.05%`): Es el indicador primordial de astringencia noble y longevidad.
2. **Prolina** (`15.77%`): Determina el grado de maduración fenólica alcanzado por la uva.
3. **Intensidad de Color** (`15.51%`): Capacidad pigmentaria y concentración de antocianos.
4. **Tonalidad / Matiz** (`11.57%`): Estadio evolutivo del vino.
5. **Absorbancia OD280/OD315** (`11.02%`): Pureza y estabilidad coloidal de la muestra.

---

## Evaluación de Calidad Enológica

Además del origen, la aplicación evalúa la muestra mediante una fórmula enológica calibrada:

$$\text{Puntaje} = \text{Base} + S_{\text{flavonoides}} + S_{\text{prolina}} + S_{\text{pureza OD}} + S_{\text{color}} + S_{\text{alcohol}} - \text{Penalización}_{\text{acidez excesiva}}$$

### Escala de Calidad:
- 🥇 **90 - 100 pts | Gran Reserva / Calidad Excepcional**: Vinos con perfil tánico sobresaliente, óptima madurez y capacidad de guarda superior a 10 años.
- 🥈 **80 - 89 pts | Reserva / Muy Buena Calidad**: Gran balance, taninos redondeados y armonía entre acidez y grado alcohólico.
- 🥉 **70 - 79 pts | Crianza / Calidad Buena**: Perfil vivaz y equilibrado, ideal para consumo en el corto y medio plazo.
- 🍷 **< 70 pts | Vino de Mesa / Joven Ligero**: Vinos frescos de menor concentración polifenólica para consumo inmediato.

---

## Instalación y Puesta en Marcha

### 1. Prerrequisitos
- Tener instalado **Python 3.10** o superior.
- Git (opcional).

### 2. Clonar o acceder a la carpeta del proyecto
```bash
cd Proyecto_vinos
```

### 3. Crear y activar el entorno virtual
En **Windows**:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

En **Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 5. Entrenar y evaluar el modelo
Ejecuta el script analítico para ver el reporte detallado en terminal y generar los archivos del dataset y el modelo:
```bash
python modelo_vinos.py
```

### 6. Iniciar la aplicación web
```bash
python app.py
```
Abre tu navegador en: **`http://127.0.0.1:5000`**

---

## 💻 Uso de la Aplicación Web

La interfaz cuenta con un diseño estético inspirado en bodegas de alta gama y laboratorios enológicos:

1. **Selector Principal del Catálogo de Bodega:**
   - Menú desplegable principal con muestras reales predefinidas: *Barolo Gran Reserva*, *Barolo Tradicional*, *Grignolino Fresco*, *Grignolino Floral*, *Barbera Clásica*, *Barbera de Capa Intensa* y *Promedio General*.
   - Al seleccionar cualquier opción, todos los 13 campos se cargan automáticamente sin necesidad de escribir números manuales.
2. **Selectores de Opciones Predeterminadas por Variable:**
   - Cada uno de los 13 parámetros cuenta con su propio desplegable de **Opciones Sugeridas** (ej. en Grado Alcohólico: *Ligero (11.8%)*, *Equilibrado (12.8%)*, *Estructurado (13.7%)*, *Potente (14.4%)*).
   - El usuario puede elegir un nivel predeterminado con un solo clic o ajustar manualmente el valor numérico con precisión decimal.
3. **Formulario Modular:** Dividido en 4 tarjetas temáticas (Grado y Acidez, Polifenoles, Color y Espectrometría, Minerales).
4. **Tarjeta de Resultados:**
   - Denominación de origen predicha, cepa y región de Italia.
   - Medalla y puntaje de calidad (0 a 100).
   - Barras de probabilidad porcentual del modelo.
   - Factores clave analizados.
   - Ficha de cata, notas de guarda y maridaje culinario recomendado.
5. **Glosario Técnico Desplegable:** Tabla con la explicación enológica de los 13 parámetros.

---

## API REST (Endpoint JSON)

La aplicación incluye un endpoint HTTP POST para realizar predicciones programáticas:

- **URL:** `http://127.0.0.1:5000/api/predecir`
- **Método:** `POST`
- **Content-Type:** `application/json`

### Ejemplo de Petición (cURL):
```bash
curl -X POST http://127.0.0.1:5000/api/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.74,
    "malic_acid": 2.01,
    "ash": 2.46,
    "alcalinity_of_ash": 17.04,
    "magnesium": 106.34,
    "total_phenols": 2.84,
    "flavanoids": 2.98,
    "nonflavanoid_phenols": 0.29,
    "proanthocyanins": 1.90,
    "color_intensity": 5.53,
    "hue": 1.06,
    "od280/od315_of_diluted_wines": 3.16,
    "proline": 1115.71
  }'
```

### Ejemplo de Respuesta:
```json
{
  "exito": true,
  "origen_predicho": "Cultivar 1 (Tipo Barolo - Piamonte DOCG)",
  "variedad_cepa": "Uva Nebbiolo",
  "region": "Región de Piamonte (Langhe / Barolo), Italia",
  "indice_calidad": 96.3,
  "categoria_calidad": "Gran Reserva / Calidad Excepcional",
  "medalla": "Oro",
  "probabilidades": [
    {
      "clase": 0,
      "nombre": "Cultivar Barolo",
      "porcentaje": 99.98,
      "es_ganador": true
    },
    {
      "clase": 1,
      "nombre": "Cultivar Grignolino",
      "porcentaje": 0.02,
      "es_ganador": false
    },
    {
      "clase": 2,
      "nombre": "Cultivar Barbera",
      "porcentaje": 0.0,
      "es_ganador": false
    }
  ]
}
```

---

## Glosario de Variables Químicas

1. **Grado Alcohólico (`alcohol`):** Porcentaje de etanol en volumen (% vol). Aporta calidez y sensación de volumen en boca.
2. **Ácido Málico (`malic_acid`):** Ácido málico en g/L. Responsable de la acidez fresca y viva.
3. **Cenizas Totales (`ash`):** Residuo mineral inorgánico (g/L) tras calcinación. Indica la mineralidad del suelo.
4. **Alcalinidad de Cenizas (`alcalinity_of_ash`):** Capacidad de neutralización ácida (meq/L).
5. **Magnesio (`magnesium`):** Mineral (mg/L) fundamental para la actividad enzimática fermentativa.
6. **Fenoles Totales (`total_phenols`):** Suma total de compuestos fenólicos antioxidantes (mg/L).
7. **Flavonoides (`flavanoids`):** Polifenoles que aportan taninos, estructura, textura aterciopelada y color.
8. **Fenoles No Flavonoides (`nonflavanoid_phenols`):** Ácidos hidroxicinámicos y derivados de la madera (mg/L).
9. **Proantocianidinas (`proanthocyanins`):** Taninos condensados estabilizadores del color (mg/L).
10. **Intensidad de Color (`color_intensity`):** Densidad óptica acumulada a 420, 520 y 620 nm.
11. **Matiz / Tonalidad (`hue`):** Relación A420/A520. Informa sobre la evolución y juventud cromática del vino.
12. **Absorbancia OD280/OD315 (`od280/od315_of_diluted_wines`):** Ratio espectrofotométrico que mide la pureza proteica.
13. **Prolina (`proline`):** Aminoácido libre mayoritario (mg/L); índice directo de madurez de la uva en cosecha.

---

## Conclusión Académica

Este proyecto demuestra cómo la combinación de algoritmos de ensamble (**Random Forest**), preprocesamiento riguroso de datos con **Pandas/Scikit-Learn** y una interfaz web intuitiva y estética desarrollada en **Flask** permite convertir datos químicos enológicos complejos en información clara, accesible y de alto valor para sommeliers, enólogos y evaluadores académicos.
