# Reconocimiento de emociones faciales utilizando Deep Learning

Este repositorio contiene una copia de todo el material relacionado con el Trabajo de Fin de Grado titulado "Reconocimiento de emociones faciales utilizando Deep Learning", realizado por:
* Luis Morales Júlvez
* Jaime Costas Insua
* Francisco Calvo González

El repositorio está organizado en las siguientes carpetas:
* `modelos`. Todos los modelos entrenados. Cada uno contiene:
  - Un archivo .py con la implementación del modelo.
  - Un archivo modelo.ckpt con los pesos del modelo.
* `imagenes`. Una serie de imágenes con las que probar los modelos.
* `scripts`. Todos los scripts necesarios para entrenar y ejecutar los modelos.

Para poder clonar el repositorio correctamente, es necesario haber instalado la extensión Git-LFS. Dicha extensión puede descargarse a través de [este enlace](https://git-lfs.com/).

## Ejecución

### Requisitos
Para poder ejecutar los modelos es necesario:
* Contar con un dataset. Todos los modelos han sido entrenados con AffectNet-7 [[1]](#1). Cualquier dataset utilizado deberá tener 7 clases: Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger. Las imágenes deben estar ordenadas en carpetas cuyo nombre corresponde con la clase a la que pertenecen. Las imágenes deben contener únicamente una cara.
* Tener instalado Python 3.11.7 o superior.
* Haber instalado las siguientes dependencias a través de PiP:
  - `pytorch`
  - `lightning`
  - `rich`
  - `torchvision`
  - `schedulefree`
### Instrucciones
Para entrenar un modelo, ejecuta el siguiente comando:
```
python3 scripts/train.py <ruta del set train> <ruta del modelo> <archivo del modelo> <epochs>
```

Para probar un modelo, ejecuta el siguiente comando:
```
python3 scripts/ejecucion.py <ruta del dataset> <archivo del modelo> <archivo de checkpoint>
```

Para generar una matriz de confusión de un modelo, ejecuta el siguiente comando:
```
python3 scripts/mat_confusion.py <ruta del dataset> <archivo del modelo> <archivo de checkpoint>
```

Algunos ejemplos de comandos:
```
python3 scripts/mat_confusion.py datasets/test vgg10a/vgg10a.py vgg10a/modelo.ckpt
python3 scripts/train.py datasets/train vgg10a vgg10a/vgg10a.py 50
python3 scripts/ejecucion.py modelos/vgg14/vgg14.py modelos/vgg14/modelo.ckpt imagenes/rabia_1.png
```

## Referencias

<a id="1">[1]</a> A. Mollahosseini; B. Hasani; M. H. Mahoor, "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild," in _IEEE Transactions on Affective Computing_, 2017.