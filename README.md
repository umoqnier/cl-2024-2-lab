# Laboratorio de Lingüística Computacional

Repositorio para entrega de prácticas del laboratorio de la clase de
Lingüistica computacional, Ciencias, UNAM

## Entregas

- Entregas en tiempo se calificarán sobre **10**
- Entregas retrasadas hasta una semana se calificarán sobre **8**
- Entregas retrasadas mas de una semana se calificará sobre **6**

### Práctica 1 — Fonética

#### Fecha de entrega: domingo 3 de marzo 2024 a las 11:59 p.m.

**Actividades**

1. Agrega un nuevo modo de búsqueda donde se extienda el comportamiento básico del buscador para ahora buscar por frases.
2. Agregar un modo de búsqueda donde dada una palabra te muestre sus homofonos[1].
    Debe mostrar su representación IPA y la lista de homofonos (si existen)
3. Observe las distribuciones de longitud de palabra y de número de morfemas por palabra para todas lenguas.
    Basado en esos datos, haga un comentario sobre las diferencias en morfología de las lenguas

### Práctica 2 — Niveles del lenguaje

#### Fecha de entrega: domingo 10 de marzo de 2024 a las 11:59 p.m.

**Actividades**

- Implementar un etiquetador POS para el idioma otomí
    - Escenario retador de bajos recursos lingüísticos (low-resources)
    - Considerar que las feature functions **deben** cambiar (van acorde a la lengua)
    - Pueden usar bibliotecas conocidas para la implementación
- Reportar accurary, precision, recall y F1-score
- Mostrar un ejemplo de oracion etiquetada (Debe ser una oracion del conjunto de pruebas). Formato libre

**Extras**

- Implementar un HMM para la misma tarea de etiquetado POS para el otomí
- Comparar las siguientes medidas con los resultados obtenidos por el modelo CRF:
  - accuracy
  - precision
  - recall
  - F1-score
- Hacer un análisis breve de los resultados
    - ¿Qué diferencias encuentran entre trabajar con textos en español y en Otomí?
    - ¿Se obtuvieron mejores resultados que con el español?
    - ¿A qué modelo le fue mejor? ¿Porqué?

### Práctica 3 - Propiedades estadísticas del lenguaje natural

#### Fecha de entrega: domingo 17 de marzo de 2024 a las 11:59 p.m.

**Actividades**

- Comprobar si las *stopwords* que encontramos en paqueterias de *NLP* coinciden con las palabras más comúnes obtenidas en Zipf
    - Utilizar el [corpus CREA](https://corpus.rae.es/frec/CREA_total.zip)
    - Realizar una nube de palabras usando las stopwords de paqueteria y las obtenidas através de Zipf
    - Responder las siguientes preguntas:
        - ¿Obtenemos el mismo resultado? Si o no y ¿Porqué?
- Comprobar si Zipf se cumple para un lenguaje artificial creado por ustedes
  - Deberán darle un nombre a su lenguaje
  - Mostrar una oración de ejemplo
  - Pueden ser una secuencia de caracteres aleatorios
  - Tambien pueden definir el tamaño de las palabras de forma aleatoria

### Práctica 4 - Subword Tokenization

#### Fecha de entrega: domingo 24 de Marzo 11:59 p.m.

**Actividades**

- Calcular la entropía de dos textos: brown y axolotl
    - Calcular para los textos tokenizados word-level
    - Calcular para los textos tokenizados con BPE
        - Tokenizar con la biblioteca `subword-nmt`
- Imprimir en pantalla:
    - Entropía de axolotl word-base y bpe
    - Entropía del brown word-base y bpe
- Responder las preguntas:
    - ¿Aumento o disminuyó la entropia para los corpus?
        - axolotl 
        - brown
    - ¿Qué significa que la entropia aumente o disminuya en un texto?
    - ¿Como influye la tokenizacion en la entropía de un texto?

**Extras**

- Realizar el proceso de normalización para el texto en Nahuatl
- Entrenar un modelo con el texto normalizado
    - Usando BPE `subword-nmt`
- Comparar entropía, typos, tokens, TTR con las versiones:
    - tokenizado sin normalizar
    - tokenizado normalizado

### Práctica 5 - Reducción de la dimensionalidad

#### Fecha de entrega: 13 de abril 2024 11:59pm

**Actividades**

Hay varios métodos que podemos aplicar para reduccir la dimensionalidad de
nuestros vectores y asi poder visualizar en un espacio de menor dimensionalidad
como estan siendo representados los vectores.

- PCA
- T-SNE
- SVD

- Entrenar un modelo word2vec
  - Utilizar como corpus la wikipedia como en la practica
  - Adaptar el tamaño de ventana y corpus a sus recursos de computo
  - Ej: Entrenar en colab con ventana de 5 y unas 100k sentencias toma ~1hr
- Aplicar los 3 algoritmos de reduccion de dimensionalidad
    - Reducir a 2d
    - Plotear 1000 vectores de las palabras más frecuentes
- Analizar y comparar las topologías que se generan con cada algoritmo
  - ¿Se guardan las relaciones semánticas? si o no y ¿porqué?
  - ¿Qué método de reducción de dimensaionalidad consideras que es mejor?

### Práctica 6 - Evaluación de modelos de lenguaje

#### Fecha de entrega: 21 de abril de 2024

**Actividades**

- Crear un par de modelos del lenguaje usando un **corpus en español**
    - Corpus: El Quijote
        - URL: https://www.gutenberg.org/ebooks/2000
    - Modelo de n-gramas con `n = [2, 3]`
    - Hold out con `test = 30%` y `train = 70%`
- Evaluar los modelos y reportar la perplejidad de cada modelo
  - Comparar los resultados entre los diferentes modelos del lenguaje (bigramas, trigramas)
  - ¿Cual fue el modelo mejor evaluado? ¿Porqué?

### Práctica 7 - Modelos neuronales de traducción automática

#### Fecha de entrega: 5 de mayo 2024 11:59pm

**Actividades**

- Explorar los datasets disponibles en el *Shared Task de Open Machine Translation de AmericasNLP 2021*
    - [Datasets](https://github.com/AmericasNLP/americasnlp2021/tree/main/data)
    - [Readme](https://github.com/AmericasNLP/americasnlp2021/tree/main#readme)
- Crear un modelo de traducción neuronal usando OpenNMT-py y siguiendo el pipeline visto en clase
    - 0. Obtención de datos y preprocesamiento
        - Considerar que tiene que entrenar su modelo de tokenization
    - 1. Configuración y entrenamiento del modelo
    - 2. Traducción
    - 3. Evaluación
        - Reportar BLEU
        - Reportar ChrF (medida propuesta para el shared task)
        - Más info: [evaluate.py](https://github.com/AmericasNLP/americasnlp2021/blob/main/evaluate.py)
- Comparar resultados con [baseline](https://github.com/AmericasNLP/americasnlp2021/tree/main/baseline_system#baseline-results)
- Incluir el archivo `*.translated.desubword`

**Extra**

- Investigar porque se propuso la medida ChrF en el Shared Task
    - ¿Como se diferencia de BLEU?
    - ¿Porqué es reelevante utilizar otras medidas de evaluación además de BLEU?

### Práctica 8: Estrategias de generación de texto

#### Fecha de entrega: 12 de Mayo 2024 11:59p.m.**

- Construir un modelo del lenguaje neuronal a partir de un corpus en español
  - Corpus: El Quijote. URL: https://www.gutenberg.org/ebooks/2000
    - **NOTA: Considera los recursos de computo. Recuerda que en la practica utilizamos ~50k oraciones**
  - Modelo de trigramas con `n = 3`
  - Incluye informacion sobre setup de entrenamiento:
    - Dimension de embeddings
    - Dimsension de capa oculta
    - Cantidad de oraciones para entrenamiento
    - Batch size y context size
  - Incluye la liga de drive de tu modelo
- Imprima en pantalla un tres ejemplos de generacion de texto
  - Proponga mejoras en las estrategias de generación de texto vistas en la práctica
  - Decriba en que consiste la estrategia propuesta
  - Compare la estrategia de la práctica y su propuesta

**Extra**

- Visualizar en 2D los vectores de las palabras más comunes (excluir STOP WORDS)

## Apéndice

### Enlaces

- [Carpeta de Google Drive](https://drive.google.com/drive/folders/17H4P-8invqeDXu_T1RDMiuw8NA2OvzgF?usp=drive_link)
