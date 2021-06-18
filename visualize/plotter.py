

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path # needed?


class Plotter(object):
    def __init__(self, *save_dir):
        self.save_dir = Path(*save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.plot_data = []

        self.subplots = False

    def subplot(self, title=None, xlabel=None, ylabel=None, grid=False, legend=False):
        self.subplots = True
        self.plot_data.append({
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'grid': grid,
            'legend': legend,
            'plots': [],
        })
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        self.subplots = False

    def plot(self, x, y=None, label=None, title=None, xlabel=None, ylabel=None, grid=False, legend=False, linewidth=1.0):
        if y is None:
            x, y = list(range(len(x))), x
        if self.subplots:
            self.plot_data[-1]['plots'].append({
                'x': x,
                'y': y,
                'label': label,
                'linewidth': linewidth,
            })
        else:
            self.plot_data.append({
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'grid': grid,
                'legend': legend,
                'plots': [{
                    'x': x,
                    'y': y,
                    'label': label,
                    'linewidth': linewidth,
                }],
            })
    
    def heatmap (self, array, xlabel='', ylabel='', xticks=None, yticks=None, xticklabels=None, yticklabels=None,
                 title=None, file_name='heatmap', cbarlabel=None, cbar_kw={}):
        # Integrate it in with plot_data
        fig, ax = plt.subplots()
        im = ax.imshow(array)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        # ... and label them with the respective list entries
        if xlabel is not None:
            ax.set_xticklabels(xticklabels)
        if ylabel is not None:
            ax.set_yticklabels(yticklabels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom") 

        ax.set_title(title)
        fig.tight_layout()
        save_path = self.save_dir / (file_name + '.png')
        plt.savefig(save_path)

    def save(self, file_name, log=None):
        # figsize in inches, dpi = dots (pixels) per inches
        fig, axes = plt.subplots(len(self.plot_data), 
            figsize=(20, 2 * len(self.plot_data)), dpi=100)
        if len(self.plot_data) == 1:
            axes = [axes]
        for ax, data in zip(axes, self.plot_data):
            for plot in data['plots']:
                ax.plot(plot['x'], plot['y'], label=plot['label'], linewidth=plot['linewidth'])
            ax.set_title(data['title'])
            ax.set_xlabel(data['xlabel'])
            ax.set_ylabel(data['ylabel'])
            ax.set_xlim(np.min([plot['x'] for plot in data['plots']]), 
                        np.max([plot['x'] for plot in data['plots']]))
            if data['grid']: ax.grid()
            if data['legend']: ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        save_path = self.save_dir / (file_name + '.png')
        fig.savefig(save_path)
        plt.close(fig)
        self.plot_data.clear()

        if log:
            print(f'{log if type(log) is str else ""}Written {save_path}')
