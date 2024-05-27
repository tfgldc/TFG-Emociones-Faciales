from torchvision.datasets.folder import ImageFolder, default_loader
import os
import glob
import random

def get_dataset(datasetPath, train_transform, val_transform):
  print("Loading dataset")
  classes = []

  # Get all classes
  for clazz in os.listdir(datasetPath):
    path = os.path.join(datasetPath, clazz)
    if os.path.isdir(path):
      classes.append(clazz)

  # Sort by class name
  classes.sort()

  images_by_class = []
  dataset_size = 0

  for clazz in classes:
    images = []
    path = os.path.join(datasetPath, clazz)

    for image in glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.png"):
      images.append(image)

    dataset_size += len(images)
    random.shuffle(images)
    images_by_class.append(images)

  train_images = images_by_class.copy()
  val_images = []

  for i in range(len(images_by_class)):
    # 20% of the images are moved to the validation set
    cutoff = int(len(train_images[i]) * 0.8)

    val_images.append(train_images[i][cutoff:])
    train_images[i] = train_images[i][:cutoff]

  print("Training set:")
  print_dataset_distribution(dataset_size, classes, train_images)
  print("Validation set:")
  print_dataset_distribution(dataset_size, classes, val_images)

  train_dataset = CustomImageFolder(classes_by_id=classes, images=train_images, root=datasetPath, transform=train_transform)
  val_dataset   = CustomImageFolder(classes_by_id=classes, images=val_images,   root=datasetPath, transform=val_transform)
  return train_dataset, val_dataset
  
def print_dataset_distribution(dataset_size, classes, images_by_class):
  total = 0
  for i in range(len(images_by_class)):
      total += len(images_by_class[i])
  
  for i in range(len(images_by_class)):
      clazz = classes[i]
      images = len(images_by_class[i])
      print(f"- {clazz}: {images} (" + f"{images / total * 100:.2f}%)")
  print(f"Total: {total} ({total / dataset_size * 100:.2f}%)")

class CustomImageFolder(ImageFolder):
  def __init__(self, classes_by_id, images, root, transform=None, target_transform=None, loader=default_loader):
    self.classes_by_id = classes_by_id
    self.paths = images
    super().__init__(root, transform, target_transform, loader)

  def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    for id in range(len(self.classes_by_id)):
      clazz = self.classes_by_id[id]

      # Convert from our ID to Pytorch's ID, since they might not match
      pytorch_id = class_to_idx[clazz]

      # Add all images from this class
      for path in self.paths[id]:
        instances.append((path, pytorch_id))
      
    return instances