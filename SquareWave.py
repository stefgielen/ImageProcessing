import numpy as np
from matplotlib import pyplot as plt

# orde van de fourier benadering
order = [1, 3, 5, 11, 31]

# array met x-waarden
x = np.arange(start=0, stop=1.5, step=0.01)
fig, axs = plt.subplots(2, len(order), sharey=True)

#square wave genereren
square_wave = np.ones(len(x))
square_wave[int(len(x)/3) : 2*int(len(x)/3)] = -1


def fourier_series(n=0, sum=0):
    # som berekenen bij oneven n
    if (n % 2) != 0:
        sum = sum + 1 / n * np.sin(n * x * 2 * np.pi)
    return sum


def fouerier_series_Plotten(array=0):
    sum = 0
    for n in range(1, (array[-1]+1)):
        sum = fourier_series(n, sum)
        # plotten in figuur als n in de array zit
        if n in order:
            index = order.index(n)
            axs[0, index].set_title("y=sin(%s x)/%s" % (n, n))
            axs[0, index].plot(x, (np.sin(n * x * 2 * np.pi)) / n)
            axs[1, index].set_title('n= %s' % n)
            axs[1, index].plot(x, sum *1.3)
            axs[1, index].plot(x, square_wave)


fouerier_series_Plotten(array=order)

plt.show()
