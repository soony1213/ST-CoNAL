import os
import importlib
import matplotlib.pyplot as plt
import pdb

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def plot_figure(x_val, y_val, y_std, label, title='', color='', path='./'):
    fig, ax1 = plt.subplots()
    if type(x_val) == type(1):
        try:
            ax1.plot(range(x_val), y_val, label=label)
        except TypeError:
            ax1.plot(x_val, y_val, label=label)
    else:
        for i in range(len(y_val)):
            try:
                ax1.plot(range(x_val), y_val[i], label=label[i], color=color[i])
                ax1.fill_between(range(x_val), y_val[i]-y_std[i], y_val[i]+y_std[i], color=color[i], alpha=0.10, edgecolor=None)
            except TypeError:
                ax1.plot(x_val, y_val[i], label=label[i], color=color[i])
                ax1.fill_between(x_val, y_val[i] - y_std[i], y_val[i] + y_std[i], color=color[i], alpha=0.10, edgecolor=None)
    ax1.grid(True)
    # ax1.set_title(title)
    ax1.legend(loc='upper left')
    # plt.ylim(-3.5, 4.7)
    # plt.ylim(-2.2, 6.7)
    # plt.ylim(-2.7, 6.2)
    if len(label)==1:
        fig.savefig(path + '/' +label + '.png', dpi=300)
    else:
        fig.savefig(path + '/' + title + '.png', dpi=300)
    plt.close()

def plot_figure_cifar10(x_val, y_val, y_std, label, title='', color='', lim=False, lim_range=None, ylabel=None, xlabel=None, path='./'):
    fig, ax1 = plt.subplots()
    if type(x_val) == type(1):
        try:
            ax1.plot(range(x_val), y_val, label=label)
        except TypeError:
            ax1.plot(x_val, y_val, label=label)
    else:
        for i in range(len(y_val)):
            try:
                ax1.plot(range(x_val), y_val[i], label=label[i], color=color[i])
                ax1.fill_between(range(x_val), y_val[i]-y_std[i], y_val[i]+y_std[i], color=color[i], alpha=0.10, edgecolor=None)
            except TypeError:
                ax1.plot(x_val, y_val[i], label=label[i], color=color[i])
                ax1.fill_between(x_val, y_val[i] - y_std[i], y_val[i] + y_std[i], color=color[i], alpha=0.10, edgecolor=None)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    if xlabel:
        plt.xlabel(xlabel, fontdict={'fontsize':17})
    if ylabel:
        plt.ylabel(ylabel, fontdict={'fontsize':14})
    ax1.grid(True)
    
    ax1.legend(loc='upper left')
    if lim:
        if lim_range is not None:
            plt.ylim(lim_range[0], lim_range[1])
        else:
            plt.ylim(-1.3, 6.3)
    # plt.ylim(-2.2, 6.7)
    # plt.ylim(-2.7, 6.2)
    
    if len(label)==1:
        fig.savefig(path + '/' +label + '.png', dpi=300)
    else:
        fig.savefig(path + '/' + title + '.png', dpi=300)
    plt.close()

def plot_figure_cifar100(x_val, y_val, y_std, label, title='', color='', linestyle='', lim=False, lim_range=None, ylabel=None, xlabel=None, path='./'):
    fig, ax1 = plt.subplots()
    if type(x_val) == type(1):
        try:
            ax1.plot(range(x_val), y_val, label=label)
        except TypeError:
            ax1.plot(x_val, y_val, label=label)
    else:
        for i in range(len(y_val)):
            try:
                ax1.plot(range(x_val), y_val[i], label=label[i], color=color[i])
                ax1.fill_between(range(x_val), y_val[i]-y_std[i], y_val[i]+y_std[i], color=color[i], alpha=0.10, edgecolor=None)
            except TypeError:
                ax1.plot(x_val, y_val[i], label=label[i], color=color[i])
                ax1.fill_between(x_val, y_val[i] - y_std[i], y_val[i] + y_std[i], color=color[i], alpha=0.10, edgecolor=None)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    if xlabel:
        plt.xlabel(xlabel, fontdict={'fontsize':17})
    if ylabel:
        plt.ylabel(ylabel, fontdict={'fontsize':14})
    # ax1.set_title(title)
    ax1.legend(loc='upper left')
    if lim:
        if lim_range is not None:
            plt.ylim(lim_range[0], lim_range[1])
        else:
            plt.ylim(-3.15, 6.7)
    # plt.ylim(-2.2, 6.7)
    # plt.ylim(-2.7, 6.2)
    if len(label)==1:
        fig.savefig(path + '/' +label + '.png', dpi=300)
    else:
        fig.savefig(path + '/' + title + '.png', dpi=300)
    plt.close()
def make_dir(root):
    from torch.utils.tensorboard import SummaryWriter
    try:
        original_umask = os.umask(0)
        os.makedirs(root, exist_ok=True)
        writer = SummaryWriter(root)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(root):
            pass
        else:
            raise
    finally:
        os.umask(original_umask)


def save_checkpoint(file_root, file_name, state, is_best):
    filename = file_root + '/' + file_name + 'pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
