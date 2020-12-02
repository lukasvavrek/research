import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def visualize_history(fig, axs, i, history):
    history_dict = history.history
    
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(history_dict['acc']) + 1)

    # special case for a fig with a single nrow
    if axs.shape != (2,):
        axs = axs[i]

    axs[0].plot(epochs, loss_values, 'bo', label='Training loss')
    axs[0].plot(epochs, val_loss_values, 'b', label='Validation loss')
    #axs[0].title('Training and validation loss')
    #axs[0].xlabel('Epochs')
    #axs[0].ylabel('Loss')
    axs[0].legend()
    
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    acc_trend = np.polyfit(epochs, acc_values, 1)
    acc_trend = np.poly1d(acc_trend)

    val_acc_trend = np.polyfit(epochs, val_acc_values, 1)
    val_acc_trend = np.poly1d(val_acc_trend)

    axs[1].plot(epochs, acc_values, 'bo', label='Training acc')
    axs[1].plot(epochs, val_acc_values, 'b', label='Validation acc')

    axs[1].plot(epochs, acc_trend(epochs), "r")
    axs[1].plot(epochs, val_acc_trend(epochs), "g")
    
    #axs[1].title('Training and validation accuracy')
    #axs[1].xlabel('Epochs')
    #axs[1].ylabel('Accuracy')
    axs[1].legend()

def visualize_spectrogram(S, sr, hop_length, show_colorbar=True):
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    if show_colorbar:
        plt.colorbar(format='%+2.0f dB')