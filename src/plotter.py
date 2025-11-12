import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 300,
    'figure.figsize': (6.4, 4.5),
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def decision_threshold_analysis():
    # Beta = 185, buffer = 50
    thresholds = np.arange(0, 41, 5)
    thresholds[0] = 1
    accuracies = np.array([0.539, 0.713, 0.798, 0.839, 0.859, 0.875, 0.88, 0.896, 0.9]) * 100
    avg_times = np.array([49.6, 54.35, 68.84, 79.92, 90.55, 107.18, 118.34, 128.35, 129.29])
    std_times = np.array([41.56, 40.93, 44.01, 40.33, 37.23, 46.76, 47.96, 48.32, 52.14])

    fig, ax = plt.subplots(2, 1)

    ax1 = ax[0]
    color = 'tab:blue'

    ax1.set_ylabel(r'$\alpha$ / %', fontsize = 15)

    ax1.plot(thresholds, accuracies, color = color, marker = 'o', label = r'$\alpha$')
 
    ax1.set_xticks(thresholds)
    ax1.set_yticks(np.arange(0, 101, 10))

    ax1.set_ylim(bottom = 50)

    ax1.grid(True)
    # ax1.legend(loc='upper left')

    ax2 = ax[1]
    color = 'tab:red'

    ax2.set_ylabel(r'$\overline{T_c}$ / s', fontsize = 15)

    ax2.set_xlabel(r'$decision\;threshold$', fontsize = 10)

    ax2.errorbar(thresholds, avg_times, yerr = std_times, color = color, marker = 's', label = r'$\overline{T_c} (s)$', capsize = 5)
 
    ax2.set_ylim(0, max(avg_times) + max(std_times) + 10)

    ax2.grid(True)
    # ax2.legend(loc='upper left')


    fig.tight_layout()

    plt.show()

    fig.savefig('results/graphs/decision_threshold_analysis.pdf')
    fig.savefig('results/graphs/decision_threshold_analysis.png')

def descriptor_buffer_size_analysis():
    buffer_sizes = np.array([1, 10, 20, 30, 40, 50])
    accuracies = np.array([0.776, 0.865, 0.906, 0.9, 0.908, 0.896]) * 100
    avg_times = np.array([105.839, 101.735, 109.586, 116.918, 121.606, 128.352])
    std_times = np.array([15.92, 19.75, 28.99, 37.09, 39.43, 48.32])

    fig, ax = plt.subplots(2, 1)

    ax1 = ax[0]
    color = 'tab:blue'
    ax1.set_ylabel(r'$\alpha$ / %', fontsize = 15)
    ax1.plot(buffer_sizes, accuracies, color = color, marker = 'o', label = r'$\alpha$')
    ax1.set_xticks(buffer_sizes)
    ax1.set_yticks(np.arange(50, 101, 10))
    ax1.set_ylim(bottom = 70)
    ax1.grid(True)

    ax2 = ax[1]
    color = 'tab:red'
    ax2.set_ylabel(r'$\overline{T_c}$ / s', fontsize = 15)
    ax2.set_xlabel(r'$buffer\;size$', fontsize = 10)
    ax2.errorbar(buffer_sizes, avg_times, yerr = std_times, color = color, marker = 's', label = r'$\overline{T_c} (s)$', capsize = 5)
    ax2.set_ylim(70, max(avg_times) + max(std_times) + 10)
    ax2.grid(True)

    fig.tight_layout()

    plt.show()

    fig.savefig('results/graphs/descriptor_buffer_size_analysis.pdf')
    fig.savefig('results/graphs/descriptor_buffer_size_analysis.png')

def swarm_size_analysis():
    swarm_sizes = np.array([4, 6, 9, 12, 16, 20, 25, 30])
    accuracies = np.array([0.67, 0.89, 0.91, 0.91, 0.9, 0.9, 0.87, 0.89]) * 100
    avg_times = np.array([342.9, 255.1, 194.9, 161.1, 139.9, 128.4, 122.2, 117.5])
    std_times = np.array([54.6, 63.2, 55.8, 46.5, 50.9, 48.3, 52.5, 39.4])
    no_decision_percentage = np.array([0.3, 0.065, 0.0275, 0.005, 0.015, 0.0325, 0.0025, 0.005]) * 100

    fig, ax = plt.subplots(3, 1)
    ax1 = ax[0]
    color = 'tab:blue'
    ax1.set_ylabel(r'$\alpha$ / %', fontsize = 15)
    ax1.plot(swarm_sizes, accuracies, color = color, marker = 'o', label = r'$\alpha$')
    ax1.set_xticks(swarm_sizes)
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.set_ylim(bottom = 50)
    ax1.grid(True)

    ax2 = ax[1]
    color = 'tab:red'
    ax2.set_ylabel(r'$\overline{T_c}$ / s', fontsize = 15)
    ax2.errorbar(swarm_sizes, avg_times, yerr = std_times, color = color, marker = 's', label = r'$\overline{T_c} (s)$', capsize = 5)
    ax2.set_xticks(swarm_sizes)
    ax2.set_ylim(0, max(avg_times) + max(std_times) + 10)
    ax2.grid(True) 

    ax3 = ax[2]
    color = 'tab:green'
    ax3.set_ylabel(r'$\epsilon $ / %', fontsize = 15)
    ax3.plot(swarm_sizes, no_decision_percentage, color = color, marker = 'd', label = 'No Decision (%)')
    ax3.set_xticks(swarm_sizes)
    ax3.set_yticks(np.arange(0, 101, 10))
    ax3.set_ylim(bottom = 0, top = 35)
    ax3.grid(True)
    ax3.set_xlabel(r'$swarm\;size$', fontsize = 10)

    fig.tight_layout()
    plt.show()

    fig.savefig('results/graphs/swarm_size_analysis.pdf')
    fig.savefig('results/graphs/swarm_size_analysis.png')  

def plot_opinions_dynamics():
    combined_info_number_of_different_opinions = np.array([1, 2.8, 8.4, 13.1, 13.2, 12.4, 9.4, 4.7, 2.5, 1.4, 1.166666667, 1, 1]) 
    only_personal_number_of_different_opinions = np.array([1.1, 3.1, 8.9, 13.2, 14.5, 15, 14.8, 14.6, 14.2, 13.1, 12.3, 12, 11.7, \
                                                            11.1, 10.8, 10.2, 9.8, 9.2, 8.9, 8.6, 8.3, 7.8, 7.5, 7.3, 6.9, 6.8, 6.6, \
                                                                6.3, 5.9, 5.7, 5.6, 5.3, 5, 4.9, 4.7, 4.5, 4.5, 4.1, 3.9, 3.8, 3.7, 3.6,\
                                                                    3.6, 3.5, 3.5, 3.5, 2.8, 2.7, 2.7, 2.75, 2.75, 2.625, 2.625, 2.5, 2.5,\
                                                                      2.25, 2.125, 1.5, 1.5, 1.28, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    number_of_dm_loops_only_personal= np.arange(1, len(only_personal_number_of_different_opinions) + 1)
    number_of_dm_loops_combined = np.arange(1, len(combined_info_number_of_different_opinions) + 1)

    plt.figure()

    plt.plot(
            number_of_dm_loops_combined,
            combined_info_number_of_different_opinions,
            color = 'tab:blue',
            marker = 'o',
            markersize = 3,
            label = r'$cooperative$'
        )
    plt.plot(
            number_of_dm_loops_only_personal,
            only_personal_number_of_different_opinions,
            color = 'tab:red',
            marker = 's', 
            markersize = 3,
            label = r'$non-cooperative$'
            )
    
    plt.xlabel(r'$DM\;loops$', fontsize = 10)
    plt.ylabel(r'$O^\ast$', fontsize = 15)

    plt.xticks(np.arange(0, 80, 5))
    plt.yticks(np.arange(0, 16, 1))

    plt.ylim(bottom = 0)
    plt.grid(True)
    plt.legend(fontsize = 10)
    plt.tight_layout()

    plt.savefig('results/graphs/opinions_dynamics.pdf')
    plt.savefig('results/graphs/opinions_dynamics.png')

    plt.show()
                          

if __name__ == "__main__":
    # decision_threshold_analysis()
    # descriptor_buffer_size_analysis()
    # swarm_size_analysis()
    plot_opinions_dynamics()