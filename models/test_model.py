#!/usr/bin/env python

import ipdb
import matplotlib.pyplot as plt
import time

from frontier import MetaethnicFrontierModel
from frontier import plot_model


def main():
    model = MetaethnicFrontierModel.empty_model(51)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_figheight(8.49)
    # fig.set_figwidth(17.56)
    # ax2.set_title(f"Asabiya")
    generation = 0

    while True:
        # plot_model(model, ax1, ax2)
        generation += 1
        # ax1.set_title(f"Generation: {generation}")

        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # plt.draw()
        # plt.show()
        
        model.step()
        print(f"Generation: {generation}")
        # plt.pause(0.05)

        if generation > 25:
            break


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()