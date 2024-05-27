import os
import sys
import shutil
import importlib

from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import CSVLogger

from checkpoints import checkpoint
from dataset import get_dataset

WEIGHTS = [ 11.410, 54.210, 44.513, 2.112, 3.792, 11.151, 20.150 ]

if __name__ == "__main__":
		if len(sys.argv) < 5:
				print("Uso: python3 entrenar.py <ruta del set train> <ruta del modelo> <archivo del modelo> <epochs>")
				exit(1)

		train_dataset_dir = sys.argv[1]
		model_dir = sys.argv[2]
		model_python_file = sys.argv[3]

		epochs = int(sys.argv[4])

		logging_dir = os.path.join(model_dir, "lightning_logs")

		checkpoints_dir = os.path.join(model_dir, "checkpoints")
		checkpoint_file, epoch_id = checkpoint(checkpoints_dir, logging_dir, epochs)

		if not os.path.isfile(model_python_file):
				print(f"El archivo {model_python_file} no existe")
				exit(1)

		if not os.path.isdir(train_dataset_dir):
				print(f"El directorio {train_dataset_dir} no existe")
				exit(1)

		if checkpoint_file is not None and not os.path.isfile(checkpoint_file):
				print(f"El archivo {checkpoint_file} no existe")
				exit(1)

		# Cargamos el modelo
		spec = importlib.util.spec_from_file_location("modelo", model_python_file)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		sys.modules["modelo"] = module


		# Preparamos el directorio
		os.makedirs(model_dir, exist_ok=True)
		os.makedirs(logging_dir, exist_ok=True)
		os.makedirs(checkpoints_dir, exist_ok=True)

		# Definir las transformaciones
		train_transform = transforms.Compose([
				transforms.Grayscale(num_output_channels=1),
				transforms.Resize((128, 128)),
				transforms.RandomRotation(20),
				transforms.RandomAffine(0, translate=(0.1, 0.1)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
		])
		val_transform = transforms.Compose([
				transforms.Grayscale(num_output_channels=1),
				transforms.Resize((128,128)),
				transforms.ToTensor()
		])

		# Cargar el dataset de entrenamiento y validación con la misma distribución de clases
		train_dataset, val_dataset = get_dataset(train_dataset_dir, train_transform, val_transform)

		# Definir el tamaño del lote
		batch_size = 64

		train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=11)
		val_loader   = DataLoader(val_dataset,                 batch_size=batch_size, num_workers=11)

		model = module.Modelo(weights = WEIGHTS)

		callbacks = [
			ModelCheckpoint( # Tiene que ser el primero!!
				monitor        = "val_loss",
				mode           = "min",
				dirpath        = checkpoints_dir,
				filename       = "expression_classifier_{epoch:02d}",
				every_n_epochs = 1,
				save_top_k     = -1,
			),
			RichProgressBar(theme=RichProgressBarTheme(metrics_format='.3e', metrics_text_delimiter="\n"), leave=True),
			LearningRateMonitor(logging_interval='step')
		]

		if checkpoint_file is not None:
				print(f"Resumiendo entrenamiento desde el checkpoint {checkpoint_file}")

		trainer = L.Trainer(
				accelerator       = "auto",
				devices           = "auto",
				strategy          = "auto",
				callbacks         = callbacks,
				max_epochs        = epochs,
				default_root_dir  = model_dir,
				log_every_n_steps = 1,
				logger            = CSVLogger(model_dir),
		)
		
		trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_file)

		print("Training completado")
