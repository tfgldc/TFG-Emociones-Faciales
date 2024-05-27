import os
import shutil

def checkpoint(checkpoints_dir, logging_dir, epochs):
		checkpoint_file = None
		highest_checkpoint = 0
		if os.path.isdir(checkpoints_dir):
				checkpoints = os.listdir(checkpoints_dir)
				if len(checkpoints) > 0:
						epoch_numbers = [int(file.split('=')[1].split('.')[0]) for file in checkpoints if file.startswith('expression_classifier_epoch=')]
						highest_checkpoint = max(epoch_numbers)

						if epochs - 1 <= highest_checkpoint:
								print(f"El último entrenamiento en este modelo ha hecho {highest_checkpoint + 1} epochs.")
								while True:
										respuesta = input("Si continúas, se eliminará el modelo y se volverá a empezar. ¿Quieres continuar? [S/N]:")
										if respuesta == "n" or respuesta == "N":
												exit(0)
										if respuesta == "s" or respuesta == "S":
												print("Eliminando modelo...")
												shutil.rmtree(checkpoints_dir)
												shutil.rmtree(logging_dir)
												break
						else:
								print(f"El último entrenamiento de este modelo ha hecho {highest_checkpoint + 1} epochs.")
								while True:
										respuesta = input("Si quieres, puedes continuar desde el último checkpoint en vez de volver a empezar. ¿Quieres resumir el entrenamiento? [S/N]:")
										if respuesta == "n" or respuesta == "N":
												print("Eliminando modelo...")
												shutil.rmtree(checkpoints_dir)
												shutil.rmtree(logging_dir)
												break
										if respuesta == "s" or respuesta == "S":
												print(f"Continuando desde el epoch {highest_checkpoint + 1}...")
												ckpt_id = f"0{highest_checkpoint}" if highest_checkpoint < 10 else f"{highest_checkpoint}"
												checkpoint_file = os.path.join(checkpoints_dir, f"expression_classifier_epoch={ckpt_id}.ckpt")
												break
		return checkpoint_file, highest_checkpoint
