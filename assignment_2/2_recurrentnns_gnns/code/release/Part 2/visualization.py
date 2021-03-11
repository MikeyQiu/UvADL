import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from more_itertools import chunked

if __name__ == "__main__":
    acc = np.load("model/accuracy.npy")
    acc_avg=[sum(x) / len(x) for x in chunked(acc, 100)]
    loss = np.load("model/loss.npy", allow_pickle=True)
    loss_avg=[sum(x) / len(x) for x in chunked(loss, 100)]
    x=range(len(acc))
    x2=range(1,len(acc),100)
    fig, ax = plt.subplots()
    #acc_plot = plt.plot(x, acc, 'steelblue', label='type1')
    #acc_avg = plt.plot(x2, acc_avg, 'blue', label='type2')


    # plt.grid(linestyle='-.')
    # plt.legend(["accuracy","averaged accuracy( per 100 iterations)"],loc=4)
    # plt.ylabel("Accuracy")
    # plt.xlabel("Iteration")
    # ax.set_title('Accuracy curve on book Democracy in US')
    # plt.savefig("Generate_acc.png")
    # plt.show()

    loss_plot = plt.plot(x, loss, 'indianred', label='type1',alpha=0.5)
    loss_avg = plt.plot(x2, loss_avg, 'red', label='type2')
    plt.grid(linestyle='-.')
    plt.legend(["Loss","averaged loss( per 100 iterations)"])
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    ax.set_title('Loss curve on book Democracy in US')
    plt.savefig("Generate_loss.png")
    plt.show()
    # #
