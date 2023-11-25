# Redes Convolucionales - Tarea 2

Implementación de un modelo de clasificación de imágenes simples ([**CIFAR-10**](https://huggingface.co/datasets/cifar10)), observando los mapas característicos generados por las diferentes capas de la red y estudiando los resultados.

## Descripción
Se emplea `PyTorch` para la creación de la red neuronal, e importar el dataset, y luego poder aplicar un *transformer* para convertir las imagenes de **PIL** a *tensor*.
En un inicio se verifica que el *dataset* sea correcto en términos generales. Empleando `mathplotlib` y `pandas` se hace una verificación visual rápida de 10 imágenes aleatorias, y luego se verifica que en el *dataset* de entrenamiento haya efectivamente 5000 imágenes por clase.
Se emplean lotes de 64 datos, sin aplicar aleatorización.
### Arquitectura de la red
Se tiene un número de capas del tipo `Conv2d`, cuya entrada empieza en el número de canales de entrada (3 por defecto), y luego va creciendo en orden de $n_{filters} 2^{i}$, para la i-ésima capa.
Además, se tiene una capa de linealización que recibe $16(2^{n_{layers} - 1})n_{filters}$, que se aplicará luego de *aplanar* la salida de las capas de convolución, para luego aplicar *dropout* para prevenir el sobreajuste.
Finalmente, se tiene otra capa de transformación lineal como salida.
Como la red recibe la información en lotes, no se almacenan las pérdidas en cada *backpropagation*, sino que se acumulan para luego almacenar el promedio. Lo mismo para el parámetro de *accurate* basándose en el número de aciertos versus el número de datos.
Además, el *forward* tiene la posibilidad de mostrar la imagen procesada en cada salida de capa  de convolución. 
> Se intentó mostrar el efecto de cada filtro, pero no se encontró una manera factible de hacerlo con `PyTorch`

Al finalizar, se exporta el modelo para analizarlo en otro cuaderno.

### Análisis de resultados
En el archivo de `evaluate` se importa el modelo entrenado, y se importa el *dataset* de prueba, en lugar de usar el de entrenamiento.

Luego, se procede a crear la matriz de confusión, iterando sobre cada elemento del *dataset*, y haciendo un *forward* con la imagen, y comparando el resultado de la red con la etiqueta de la imagen, y sumando 1 al elemento de la matriz de confusión correspondiente.

Para los primeros 10 elementos se muestra la salida de cada capa de convolución, y la imagen, junto con su etiqueta y predicción.

De aquí se pueden notar varias cosas:
* De los **automóviles**, en general suele detectar la parte baja y la forma de las ruedas en la carrocería.
* De un **barco**, se fija en la forma alargada y *lisa* del mismo, sin alas o protuberancias.
* Del **avión** pareciera haber aprendido a identificarlos con la parte de abajo y el timón.
* De los **caballos** pareciera haber aprendido que son cuadrúpedos y que generalmente están de pie, con el lomo en la parte superior de la imagen. Por esto es que se puede entender que la imagen del gato acostado con las patas abiertas, sea interpretado como un caballo.
* Del **gato** pareciera haber aprendido a diferenciar las orejas y la forma de la nariz. Ninguna de las dos cosas están en la imagen del gato acostado, por lo que es comprensible que se interprete antes como un caballo.
* Del **camión**, se entiende que toma en cuenta su altura y forma cuadrada.

Del resto de elementos no queda claro a qué cosas aprendió a identificar, sin embargo, aprendió lo suficiente para tener una precisión de **69.71%**
Además, la matriz de confusión parece bastante diagonalizada, lo que supone un gran número de aciertos en comparación con los fallos.
Finalmente, al analizar la precisión de las clases por separado, se tuvo que la red identifica mucho más fácil un automóvil, una rana, un barco y un camión, que un pájaro, un gato y un ciervo. El resto está bastante apegado ala media general.