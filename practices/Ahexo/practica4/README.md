# Práctica 4: Tokenización
Implementada por Alejandro Axel Rodríguez Sánchez (@Ahexo)  
Lingüística computacional (2024-2, 7014)  
Facultad de Ciencias UNAM  

## Cómo ejecutar

1. Se recomienda generar un entorno virtual de Python (al menos 3.9).
```sh
python3 -m venv practica4
source practica4/bin/activate
```

2. Instalar las bibliotecas requeridas, se puede usar el archivo `requirements.txt` para esto:
```sh
pip install -r requirements.txt
```

3. Ejectuar el script `practica4.py`.
```sh
python practica4.py
```

## Cuestionamientos

> ¿Aumentó o disminuyó la entropía de los corpus [Luego de tokenizar con BPE]?

La entropía de ambos disminuyó luego de aplicar la tokenización con BPE, aproximadamente en un 30%:

| Corpus   | Lengua   |   Entropía (Word-Level) |   Entropía (BPE) |
|----------|----------|-------------------------|------------------|
| Axolotl  | Náhuatl  |                 11.4922 |          8.35031 |
| Brown    | Inglés   |                 10.6386 |          8.35453 |

Porcentaje de reducción para Axolotl:  
$(11.4922)x = 8.35031$  
$x \approx 72.66 \% $


Porcentaje de reducción para Brown:  
$(10.6386)x = 8.35453$  
$x \approx 78.53 \% $


> ¿Qué significa que la entropia aumente o disminuya en un texto?

Que este es menos o más predecible. Generalmente, una entropía alta es sinónimo de un texto con una variedad de tipos muy amplia, lo que se traduce en un texto mas errático. Al bajar la entropía, se entiende que hemos aplicado algún método para reducir estos últimos, de modo que el texto se vuelve más predecible.

> ¿Como influye la tokenizacion en la entropía de un texto?

En que la disminuye, pues a menos tipos que tokens, el corpus se vuelve más suceptible a predicciones.

### Sobre la evaluación con el corpus de náhuatl normalizado

Los resultados son ligeramente mejores luego de efectuar una normalización del corpus del náhuatl, sin cambios drásticos.

Porcentaje de reducción para Axolotl Normalizado:  
$(11.1388)x = 8.24771$  
$x \approx 74.04 \% $  

La reducción de entropía mejora apenas un 2% respecto con el corpus sin normalizar.

Los tokens si vieron cambios mas sustanciales, lo cual es comprensible dado que la normalización suele modificar fuertemente las grafías del texto.

| Número | Normalizado | Original |
|--------|--------|--------|
| 1      | in     | yn     |
| 2      | ki@@   | in     |
| 3      | i@@    | i@@    |
| 4      | tla@@  | qui@@  |
| 5      | ka@@   | tla@@  |
| 6      | ti@@   | a@@    |
| 7      | a@@    | ti@@   |
| 8      | mo@@   | o@@    |
| 9      | o@@    | .      |
| 10     | s      | mo@@   |
| 11     | j@@    | te@@   |
| 12     | te@@   | ca@@   |
| 13     | .      | ,      |
| 14     | l@@    | to@@   |
| 15     | to@@   | y@@    |