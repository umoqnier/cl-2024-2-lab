# Práctica 2: Niveles lingüísticos II
Implementada por Alejandro Axel Rodríguez Sánchez (@Ahexo)  
Lingüística computacional (2024-2, 7014)  
Facultad de Ciencias UNAM  

## Cómo ejecutar
1. Se recomienda generar un entorno virtual de Python (al menos 3.9).
```sh
python3 -m venv practica2
source practica2/bin/activate
```

2. Instalar las bibliotecas requeridas, se debe usar el archivo `requirements.txt` para esto:
```sh
pip install -r requirements.txt
```

3. Ejectuar el script `practica2.py`.
```sh
python practica2.py
```

## Cuestionamientos
> ¿Qué diferencias encuentran entre trabajar con textos en español y en Otomí?

La variedad de corpus y herramientas pre-entrenadas con las que contamos en el español es mayor, el otomí, al ser una lengua mas marginal, cuenta con muchos menos recursos y referencias, lo que hace mas engorroso trabajar con el.

> ¿Se obtuvieron mejores resultados que con el español?

Si, los resultados estan mas enriquecidos y es mas fácil definir *features* para el modelo.

> ¿A qué modelo le fue mejor? ¿Porqué?

Los resultados con el modelo en español son ligeramente mejores, esto podría deberse a las razones previamente esbozadas o que nuestro modelo de otomí tiene una robustez que compensa sus deficiencias.