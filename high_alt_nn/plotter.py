from csv import reader
import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
def plot_mse(name, csv_file, identifier, save_dir_i):
    plt.figure(figsize=(22,10))
    losses = []
    maes = []
    val_losses = []
    val_maes = []
    with open(csv_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        if header !=None:
            for row in csv_reader:
                losses.append(row[1])
                maes.append(row[2])
                val_losses.append(row[3])
                val_maes.append(row[4])
    print(val_losses)
    print(losses)
    num_epochs = len(losses)
    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), val_losses, linestyle='-', label= 'Val Loss', color='orange')
    ax.plot(range(num_epochs), losses, linestyle='-', label = 'Train Loss', color='blue')
    plt.title('Val and Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))

    plt.savefig(save_dir_i + name +'_mse_plot.png')
    plt.close()

if __name__ == '__main__':
    plot_mse('test', 'train.csv', '', os.path.join(os.getcwd(), 'loss_graphs/'))
