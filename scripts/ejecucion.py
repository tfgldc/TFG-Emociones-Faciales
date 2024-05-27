import sys
import os
import importlib

from torchvision import transforms
from torch import Tensor, stack
from typing import Optional, Callable
from PIL import Image

WEIGHTS = [ 11.410, 54.210, 44.513, 2.112, 3.792, 11.151, 20.150 ]
LABELS = [ "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise" ]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 ejecucion.py <archivo del modelo> <archivo de checkpoint> <imagen>")
        exit(1)

    model_python_file = sys.argv[1]
    model_dir = os.path.dirname(os.path.realpath(model_python_file))
    dir_ckpt = sys.argv[2]
    image_path = sys.argv[3]

    # Cargamos el modelo
    spec = importlib.util.spec_from_file_location("modelo", model_python_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["modelo"] = module

    # Cargamos la imagen
    image = Image.open(image_path)

    val_transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((128, 128)),
		transforms.ToTensor()
	])

    # Aplicamos las transformaciones
    input = val_transform(image).unsqueeze(0)

    model = module.Modelo.load_from_checkpoint(dir_ckpt, weights = WEIGHTS)
    model.eval()

    prediction = model(input).data.cpu().numpy()[0]
    prediction = LABELS[prediction.argmax()]

    print("Predicción:", prediction)