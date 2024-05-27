from matplotlib import pyplot as plt
import sys
import pandas as pd

def pintar_grafica():
    acc = pd_union['train_accu_epoch']
    val_acc = pd_union['val_accu']

    loss = pd_union['train_loss_epoch']
    val_loss = pd_union['val_loss']

    epochs_range = range(len(pd_union))

    plt.figure(figsize=(10, 8))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig("training_validation_accuracy.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Loss')
    plt.savefig("training_validation_loss.png")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 metricas.py <archivo .csv del entrenamiento>")
        exit(1)

    df = pd.read_csv(sys.argv[1])

    df_train = df[df['val_accu'].isna()]
    df_val = df[~df['val_accu'].isna()]

    columns_to_drop = [col for col in df_train.columns if col.endswith('step') or col.startswith('val')]
    df_t = df_train.drop(columns=columns_to_drop, inplace=False)
    df_t = df_t.dropna()
    df_t.set_index('epoch', inplace=True)

    columns_to_drop = [col for col in df_val.columns if col.endswith('_epoch') or col.startswith('train') or col.startswith('step')]
    df_v = df_val.drop(columns=columns_to_drop , inplace=False)
    df_v.set_index('epoch', inplace=True)

    pd_union = pd.merge(df_t, df_v, on='epoch', how='outer')

    pintar_grafica()