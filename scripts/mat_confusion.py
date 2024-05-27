import sys
import os
import importlib
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
from torch import Tensor, stack
import lightning as L
from torch.utils.data import DataLoader
from itertools import chain
import matplotlib.pyplot as plt

WEIGHTS = [ 11.410, 54.210, 44.513, 2.112, 3.792, 11.151, 20.150 ]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 mat_confusion.py <ruta del dataset> <archivo del modelo> <archivo de checkpoint>")
        exit(1)

    train_dataset_dir = sys.argv[1]
    model_python_file = sys.argv[2]
    model_dir = os.path.dirname(os.path.realpath(model_python_file))
    dir_ckpt = sys.argv[3]
    cf_path = os.path.join(model_dir, "confusion_matrix.png")

    spec = importlib.util.spec_from_file_location("modelo", model_python_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["modelo"] = module

    val_transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((128, 128)),
		transforms.ToTensor()
	])

    val_dataset = ImageFolder(train_dataset_dir, transform=val_transform)
    val_loader   = DataLoader(val_dataset, batch_size=128, num_workers=11)

    model = module.Modelo(weights = WEIGHTS)

    trainer = L.Trainer(
        accelerator      = "auto",
		devices          = "auto",
		strategy         = "auto",
        default_root_dir = model_dir
    )
    model.eval()

    predictions = []
    targets = []
    predictions_batches = trainer.predict(model, val_loader, ckpt_path=dir_ckpt)
    predictions = stack(list(chain(*predictions_batches))) # merge the batches

    true_labels = []
    for images, labels in val_loader:
        true_labels.extend(labels.tolist())

    true_labels_tensor = Tensor(true_labels)

    target = true_labels_tensor

    confmat = MulticlassConfusionMatrix(num_classes=7)
    confmat.update(predictions, target)
    fig, ax = confmat.plot(labels=val_dataset.class_to_idx.keys())
    plt.savefig(cf_path)
    print("Matriz de confusión guardada en", cf_path)

    mat = confmat.compute()
    print("Matriz de confusión:")
    print(mat)

    mat = mat / mat.sum(axis=1)[:, np.newaxis]

    print("Matriz de confusión normalizada:")
    print(mat)

    print("Precisión por clase:")
    print(mat.diagonal())

    print("Precisión media:")
    print(mat.diagonal().mean())
