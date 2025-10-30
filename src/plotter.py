import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_and_time_to_threshold():
    # Beta = 185, buffer = 50
    thresholds = np.arange(0, 41, 5)
    thresholds[0] = 1
    accuracies = np.array([0.539, 0.713, 0.798, 0.839, 0.859, 0.875, 0.88, 0.896, 0.9]) * 100
    avg_times = np.array([49.6, 54.35, 68.84, 79.92, 90.55, 107.18, 118.34, 128.35, 129.29])
    std_times = np.array([41.56, 40.93, 44.01, 40.33, 37.23, 46.76, 47.96, 48.32, 52.14])

    fig, ax = plt.subplots(2, 1, figsize = (8, 6))

    ax1 = ax[0]
    color = 'tab:blue'

    ax1.set_ylabel('Accuracy')

    ax1.plot(thresholds, accuracies, color = color, marker = 'o', label = 'Accuracy')
 
    ax1.set_xticks(thresholds)
    ax1.set_yticks(np.arange(0, 101, 10))

    ax1.set_ylim(bottom = 40)

    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax[1]
    color = 'tab:red'

    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Average Time (s)')

    ax2.errorbar(thresholds, avg_times, yerr = std_times, color = color, marker = 's', label = 'Avg Time to Threshold', capsize = 5)
 
    ax2.set_ylim(0, max(avg_times) + max(std_times) + 10)

    ax2.grid(True)
    ax2.legend(loc='upper left')


    plt.title('Accuracy and Time to Threshold vs Decision Threshold (Beta=185, Buffer=50)')
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    plot_accuracy_and_time_to_threshold()