import matplotlib.pyplot as plt

def training_plot(history):
    loss = history.history['loss']
    # acc = history.history['acc']
    val_loss = history.history['val_loss']
    # val_acc = history.history['val_acc']

    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'b.')
    plt.title('Training and validation loss')
    plt.show()
