# Práctica 2

Níveles del lenguaje (II)

## Objetivo

Construir un etiquetador POS para el idioma otomí.  
Variante de la región de ...  

Dada una oración en otomí, regresar la secuencia de etiquetas gramaticales.

### Ejemplo

**Entrada**  
`ndóphu̱di dópe̱phí bit bimähtratágí ko chíkóhté`

**Salida**  
`ADJ NOUN VERB DET ADJ NOUN VERB DET ...`


## Apéndice

### Boceto de implementación

**Enfoque inmediato**  

- tokenizar
- identificar cada token con su categoria
    - lista de articulos, sustantivos, etc
    - los que sobren son verbos??

**Linear-Chain CRF**
<!-- TODO -->
