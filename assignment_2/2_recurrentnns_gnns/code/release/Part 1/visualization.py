import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    a = np.load("lstm/peep_accuracy_41_40_modifed.npy")
    b = np.load("lstm/peep_accuracy_41_42_modifed.npy")
    c = np.load("lstm/peep_accuracy_41_44_modifed.npy")
    data=[a[2000:3000],b[2000:3000],c[2000:3000]]
    mean=np.mean(data, axis=0)
    mean_of_mean=np.mean(mean)
    std=np.std(data,axis=0)
    std_of_std=np.mean(std)
    #print(std)
    print(mean_of_mean)
    print(std_of_std)
    y1=mean
    y1_1 = (mean+std)
    y1_2 = (mean-std)
    x=range(3000)
    fig, ax = plt.subplots()
    #
    #
    l1_avg = plt.plot(x, y1, 'steelblue', label='type1')
    ax.fill_between(x, y1_1, y1_2,
                    facecolor='gold',alpha=0.5)
    plt.grid(linestyle='-.')
    plt.legend(["Average accuracy","std"],loc=4)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    ax.set_title('Accuracy of peephole LSTM over binary palindrome(T=10)')
    plt.savefig("T_10_peepLSTM.png")
    plt.show()
    # #
