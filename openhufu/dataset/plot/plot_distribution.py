import matplotlib.pyplot as plt
import numpy as np

def plot_distribution_by_clients(n_clients, labels, classes, figsize=(12, 8)):
    n_classes = len(classes)
    plt.figure(figsize=figsize)
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(labels):
        for idx in idc:
            label_distribution[idx].append(c_id)

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, n_clients + 1.5, 1),
                label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                                      c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.legend(title="Class")
    plt.title("Label Distribution on Different Clients")
    plt.show()

def plot_distribution_by_classes(n_clients, labels, classes, figsize=(12, 8)):
    n_classes = len(classes)
    plt.figure(figsize=figsize)
    label_distribution = [[] for _ in range(n_clients)]
    for c_id, idc in enumerate(labels):
        for idx in idc:
            label_distribution[c_id].append(idx)

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, n_classes + 1.5, 1),
                label=[_ for _ in range(n_clients)], rwidth=0.5)
    plt.xticks(classes, ["%s" %
                                      c_id for c_id in classes])
    plt.xlabel("Class ID")
    plt.ylabel("Number of Samples")
    plt.legend(title="Client")
    plt.title("Label Distribution on Different Classes")
    plt.show()

if __name__ == "__main__":
    n_clients = 5
    classes = np.arange(10)
    labels = [np.random.choice(classes, 100).tolist() for _ in range(n_clients)]
    plot_distribution_by_clients(n_clients, labels, classes)
    plot_distribution_by_classes(n_clients, labels, classes)