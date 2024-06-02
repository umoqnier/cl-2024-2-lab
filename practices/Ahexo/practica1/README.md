# Práctica 1: Niveles lingüísticos I
Implementada por Alejandro Axel Rodríguez Sánchez (@Ahexo)
Lingüística computacional (2024-2, 7014)
Facultad de Ciencias UNAM

## Cómo ejecutar
1. Se recomienda generar un entorno virtual de Python (al menos 3.9).
```sh
python3 -m venv practica1
source practica1/bin/activate
```

2. Instalar las bibliotecas requeridas, se puede usar el archivo `requirements.txt` para esto:
```sh
pip install -r requirements.txt
```

3. Ejectuar el script `practica1.py`.
```sh
python practica1.py
```

## Detalles

Se implementaron todos los requerimientos básicos de la práctica y el extra, esto es:
    + Búsqueda de transcripciones IPA de palabras y oraciones.
    + Búsqueda de palabras homofonas.
    + Selección y descarga en buffer de los corpus para distintos idiomas.
    + Sugerencias antes palabras no encontradas.

Al ingresar, el programa solicitará al usuario seleccionar una lengua para descargar el corpus (si este no está disponible).
```
Lenguas disponibles: ['ar', 'de', 'en_UK', 'en_US', 'eo', 'es_ES', 'es_MX', 'fa', 'fi', 'fr_FR', 'fr_QC', 'is', 'ja', 'jam', 'km', 'ko', 'ma', 'nb', 'nl', 'or', 'ro', 'sv', 'sw', 'tts', 'vi_C', 'vi_N', 'vi_S', 'yue', 'zh']
Deja la casilla en blanco y presiona enter para salir.
lang>> es_MX
Corpus no encontrado. Descargando...
```

Posteriormente, el programa solicitará al usuario seleccionar el modo del que hará uso: Búsqueda de homofonos o de transcripciones.
```
Ingresa 1 para buscar homofonos, cualquier otra cosa para buscar transcripciones IPA:
modo>>1
[es_MX][Homofonos]>> 
```
```
Corpus no encontrado. Descargando...
Ingresa 1 para buscar homofonos, cualquier otra cosa para buscar transcripciones IPA:
modo>> 0
[es_MX][Transcripciones]>>
```

Ahora podemos hacer las consultas que queramos.
En el caso de la búsqueda de homofonos, el programa únicamente buscará homofonos para la primera palabra ingresada.
```
[es_MX][Homofonos]>> casa
Se buscó: casa
['/kasa/']
Homofonos hallados: ['casa', 'caza']
[es_MX][Homofonos]>> 
```

Para el caso de la búsqueda de transcripciones, se van a buscar todas las palabras de la entrada (es decir, se buscan tanto palabras sueltas como oraciones completas).
```
[es_MX][Transcripciones]>> Mi kasa está sobre un cerro
/mi/ /kasa[!]/ /eˈsta/ /soβɾe/ /un/ /sero/ 
No se encontró <<kasa>> en el dataset, se muestran algunas palabras similares:
basa: ['/basa/']
casa: ['/kasa/']
gasa: ['/gasa/']
jasa: ['/xasa/']
lasa: ['/lasa/']
masa: ['/masa/']
nasa: ['/nasa/']
pasa: ['/pasa/']
rasa: ['/rasa/']
tasa: ['/tasa/']
[es_MX][Transcripciones]>> 

```

Si dejamos la entrada del programa en blanco mientras estamos en cualquier modo, el programa regresará al inicio, donde podemos seleccionar un nuevo corpus para hacer búsquedas. Si la volvemos a dejar en blanco, el programa finalizará.

## Cuestionamientos

> 3. Observe las distribuciones de longitud de palabra y de número de morfemas por palabra para todas lenguas. Basado en esos datos, haga un comentario sobre las diferencias en morfología de las lenguas

Es claro que las distribuciones de longitud y número de morfemas por palabra varía considerablemente entre lenguas, esto se debe a que la complejidad morfológica entre las lenguas varía dependiendo de sus influencias y antecedentes.

Cuando en una lengua se forman palabras encadenando morfemas (las unidades significativas más pequeñas del lenguaje) sin cambiar su ortografía o fonética, se suele decir que se trata de una lengua **aglutinante**. Ejemplos claros están en el alemán, el japonés o el nahuatl.

En contraste, otras lenguas utilizan en su mayoría morfemas libres donde cada palabra tiende a ser independiente y expresar un único significado o función, las cuales suelen recibir el calificativo de lenguas **analíticas**. Algunos ejemplos son el inglés, el chino o el español.