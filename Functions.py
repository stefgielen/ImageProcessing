import matplotlib.pyplot as plt


def plot_figures(mainTitle, plots, titles, rowSize = 4):
    """

    :param title: Titel can de gehele plot
    :param plots: np array van te plotten figuren
    :param titles: list van titelStrings individuele plots
    """
        #nodige aantal rijen definieren
    tot = plots.shape[0]
    cols = rowSize
    rows = tot // cols
    rows += tot % cols

    position = range(1, tot + 1)
        #figuur aanmaken
    fig = plt.figure()
    for i in range(0, tot):
        ax = fig.add_subplot(rows, cols, position[i])
        ax.set_title(titles[i])
        #ax.xaxis.set_ticklabels([]);
        #ax.yaxis.set_ticklabels([])
        ax.imshow(plots[i], cmap='gray')
    fig.suptitle(mainTitle)
    plt.show()

