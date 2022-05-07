import matplotlib.pyplot as plt
import numpy as np
import os


def visualize(data, title='Image'):
    axes = AxesSequence()

    print(f'shape of {title}: {data.shape}')

    for i_class, sound_class in enumerate(data):
        for i, ax in zip(range(len(sound_class)), axes):
            spec = np.squeeze(sound_class[i])
            ax.imshow(spec)
            ax.set_title(f'sound_class: {i_class} - {title} {i}')
    axes.show()


class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""

    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self._i = 0  # Currently displayed axes index
        self._n = 0  # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8],
                               visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i+1 < len(self.axes):
            self.axes[self._i].set_visible(False)
            self.axes[self._i+1].set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i].set_visible(False)
            self.axes[self._i-1].set_visible(True)
            self._i -= 1

    def show(self):
        self.axes[0].set_visible(True)
        plt.show()


if __name__ == '__main__':
    visualize()
