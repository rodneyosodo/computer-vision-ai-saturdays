import matplotlib.pyplot as plt

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-d', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-d', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],'r-d', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"],'b-d', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)