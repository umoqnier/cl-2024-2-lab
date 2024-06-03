# Práctica 5: Reducción de la dimensionalidad
Implementada por Alejandro Axel Rodríguez Sánchez (@Ahexo)  
Lingüística computacional (2024-2, 7014)  
Facultad de Ciencias UNAM  

## Cómo ejecutar

1. Se recomienda generar un entorno virtual de Python (al menos 3.9).
```sh
python3 -m venv practica5
source practica5/bin/activate
```

2. Instalar las bibliotecas requeridas, se puede usar el archivo `requirements.txt` para esto:
```sh
pip install -r requirements.txt
```

3. Ejectuar el script `practica5.py`.
```sh
python practica5.py
```

## Detalles

Por detalles de cómputo, se recomienda ejecutar esta práctica desde el notebook de jupyter.

La biblioteca para obtener los dumps de wikipedia me estuvo dando problemas para ejecutarla en mi equipo: No podía hallarla en los repositorios de pip, y cuando sí, los dups de wikipedia no estaban disponibles, me soltaban un error 404.

Al buscar alternativas, encontré que podía descargar manualmente el contenido de los artículos de Wikipedia por medio de peticiones web. Es una aproximación un poco mas limitada (porque solo podemos operar sobre artículos de temas en particular) pero que nos da suficiente material para trabajar.

## Cuestionamientos

# Cuestionamientos

> ¿Se guardan las relaciones semánticas?

No mucho, creo que t-SNE es quien mejor las guarda, PCA y SVD muestran resultados mas parecidos.

> ¿Qué método de reducción de dimensionalidad consideras que es mejor?

Toca ser pragmáticos: PCA es muy eficiente en términos computacionales, pero se sehace de muchas relaciones semánticas. t-SNE se desempeña mejor en este último aspecto, pero es computacionalmente mas costosa, pues se divide en mas matrices. SVD es poco mas versatil, pero sufre de las mismas deficiencias de PCA, pues se deja las relaciones no-lineales detrás.

Dependerá del problema que se quiere atacar y los recursos de cómputo con los que se cuenta.
