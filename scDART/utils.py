import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



def plot_latent(z1, z2, anno1 = None, anno2 = None, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    """\
    Description
        Plot latent space
    Parameters
        z1
            the latent space of first data batch, of the shape (n_samples, n_dimensions)
        z2
            the latent space of the second data batch, of the shape (n_samples, n_dimensions)
        anno1
            the cluster annotation of the first data batch, of the  shape (n_samples,)
        anno2
            the cluster annotation of the second data batch, of the  shape (n_samples,)
        mode
            "joint": plot two latent spaces(from two batches) into one figure
            "separate" plot two latent spaces separately
        save
            file name for the figure
        figsize
            figure size
    """
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
        "markerscale": 4,
        "colormap": "Paired"
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    if mode == "modality":
        colormap = plt.cm.get_cmap("tab10")
        axs = fig.add_subplot()
        axs.scatter(z1[:,0], z1[:,1], color = colormap(1), label = "scRNA-Seq", s = _kwargs["s"], alpha = _kwargs["alpha"])
        axs.scatter(z2[:,0], z2[:,1], color = colormap(2), label = "scATAC-Seq", s = _kwargs["s"], alpha = _kwargs["alpha"])
        axs.legend(loc='upper left', frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        
        axs.tick_params(axis = "both", which = "major", labelsize = 15)

        axs.set_xlabel(axis_label + " 1", fontsize = 19)
        axs.set_ylabel(axis_label + " 2", fontsize = 19)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)  

    elif mode == "joint":
        axs = fig.add_subplot()
        cluster_types = set([x for x in np.unique(anno1)]).union(set([x for x in np.unique(anno2)]))
        cluster_types = sorted(list(cluster_types))
        colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_types))

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]
            index2 = np.where(anno2 == cluster_type)[0]
            axs.scatter(np.concatenate((z1[index,0], z2[index2,0])), np.concatenate((z1[index,1],z2[index2,1])), color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
        
        axs.legend(loc='upper left', frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        
        axs.tick_params(axis = "both", which = "major", labelsize = 15)

        axs.set_xlabel(axis_label + " 1", fontsize = 19)
        axs.set_ylabel(axis_label + " 2", fontsize = 19)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)  


    elif mode == "separate":
        axs = fig.subplots(1,2)
        cluster_types = set([x for x in np.unique(anno1)]).union(set([x for x in np.unique(anno2)]))
        cluster_types = sorted(list(cluster_types))
        colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_types))


        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]

            if index.shape[0] != 0:
                axs[0].scatter(z1[index,0], z1[index,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
        axs[0].legend(loc='upper left', frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        axs[0].set_title("scRNA-Seq", fontsize = 25)

        axs[0].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[0].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[0].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[0].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
        axs[0].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)  


        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno2 == cluster_type)[0]

            if index.shape[0] != 0:
                axs[1].scatter(z2[index,0], z2[index,1], color = colormap(i), label = cluster_type, s = _kwargs["s"], alpha = _kwargs["alpha"])
        # axs[1].axis("off")
        axs[1].legend(loc='upper left', frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = _kwargs["markerscale"])
        axs[1].set_title("scATAC-Seq", fontsize = 25)

        axs[1].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[1].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[1].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[1].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
        axs[1].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)  

    plt.tight_layout()
    if save:
        fig.savefig(save, bbox_inches = "tight")
    
    print(save)
    return fig, axs


def plot_latent_pt(z1, z2, pt1, pt2, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)

    if mode == "joint":
        ax = fig.add_subplot()
        z = np.concatenate((z1, z2), axis = 0)
        pt = np.concatenate((pt1, pt2))
        z = z[np.argsort(pt),:]
        sct = ax.scatter(z[:,0], z[:,1], c = np.arange(z1.shape[0] + z2.shape[0]), cmap = plt.get_cmap('gnuplot'), **_kwargs)
        

        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  

        cbar = fig.colorbar(sct,fraction=0.046, pad=0.04, ax = ax)
        cbar.ax.tick_params(labelsize = 20)


    elif mode == "separate":
        axs = fig.subplots(1,2)
        sct1 = axs[0].scatter(z1[:,0], z1[:,1], c = pt1, cmap = plt.get_cmap('gnuplot'), **_kwargs)
        axs[0].set_title("scRNA-Seq", fontsize = 25)

        axs[0].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[0].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[0].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[0].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
        axs[0].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)  

        cbar = fig.colorbar(sct1,fraction=0.046, pad=0.04, ax = axs[0])
        cbar.ax.tick_params(labelsize = 20)


        sct2 = axs[1].scatter(z2[:,0], z2[:,1], c = pt2, cmap = plt.get_cmap('gnuplot'), **_kwargs)
        axs[1].set_title("scATAC-Seq", fontsize = 25)

        axs[1].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[1].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[1].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[1].set_xlim(np.min(np.concatenate((z1[:,0], z2[:,0]))), np.max(np.concatenate((z1[:,0], z2[:,0]))))
        axs[1].set_ylim(np.min(np.concatenate((z1[:,1], z2[:,1]))), np.max(np.concatenate((z1[:,1], z2[:,1]))))
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)  

        cbar = fig.colorbar(sct2,fraction=0.046, pad=0.04, ax = axs[1])
        cbar.ax.tick_params(labelsize = 20)

    if save:
        fig.savefig(save, bbox_inches = "tight")
    
    print(save)


def plot_backbone(z1, z2, T, mean_cluster, groups, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)

    if mode == "joint":
        cluster_types, idx = np.unique(groups, return_index=True)
        cluster_types = cluster_types[np.argsort(idx)]
        ax = fig.add_subplot()
        cmap = plt.get_cmap('tab20', len(cluster_types))
        z = np.concatenate((z1, z2), axis = 0)
        
        for i, cat in enumerate(cluster_types):
            idx = np.where(groups == cat)[0]
            cluster = ax.scatter(z[idx,0], z[idx,1], color = cmap(i), cmap = 'tab20')
            cluster.set_label("group"+str(cat))

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if T[i,j] != 0:
                    ax.plot([mean_cluster[i, 0], mean_cluster[j, 0]], [mean_cluster[i, 1], mean_cluster[j, 1]], 'r-')
        

        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 


    if save:
        fig.savefig(save, bbox_inches = "tight")
    
    print(save)
