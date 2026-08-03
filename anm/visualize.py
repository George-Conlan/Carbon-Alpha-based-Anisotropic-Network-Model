import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D as A3
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_contact_map(adj):
    ax = matplotlib.pyplot.gca()
    im = ax.imshow(adj, cmap='binary')
    matplotlib.pyplot.colorbar(im, ax=ax)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    return ax
def plot_msf_overlay(pred_msf, exp_msf):
    residues = np.arange(len(pred_msf))
    ax = matplotlib.pyplot.gca()
    ax.plot(residues, pred_msf, label='predicted (scaled)')
    ax.plot(residues, exp_msf, label='experimental')
    ax.legend()
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('MSF')
    return ax
def plot_correlation_heatmap(cov, N):
    reduced = np.zeros((N,N))
    ax = matplotlib.pyplot.gca()
    for i in range(N):
        for j in range(N):
            reduced[i,j] = np.trace(cov[3*i:3*i+3, 3*j:3*j+3])
    diag = np.sqrt(np.diag(reduced))
    normalized = reduced/np.outer(diag,diag)
    im = ax.imshow(normalized, cmap='coolwarm', vmin=-1, vmax=1)
    matplotlib.pyplot.colorbar(im, ax=ax)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    return ax

def plot_flexibility_heatmap_1d(values, cmap='hot',vmin=None, vmax=None):
    ax = matplotlib.pyplot.gca()
    im = ax.imshow(values.reshape(1,-1), cmap=cmap, aspect='auto',vmin=vmin,vmax=vmax)
    ax.set_yticks([])
    matplotlib.pyplot.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_xlabel('Residue Index')
    return ax

def plot_flexibility_heatmap_3d(coords, values, cmap='hot',vmin=None, vmax=None, views=(30,-60)):
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    ax = matplotlib.pyplot.figure().add_subplot(projection='3d')
    ax.view_init(elev=views[0], azim=views[1])
    xs, ys, zs= coords[:,0], coords[:,1], coords[:,2]
    ax.plot(xs,ys,zs)
    sc =ax.scatter(xs,ys,zs, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
    matplotlib.pyplot.colorbar(sc,ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return ax

def animate_mode(coords, eigvec, output_path, n_frames=40, max_displacement=3.0, cmap='hot', views=(30, -60), fps=20):
    N = coords.shape[0]
    disp = eigvec.reshape(N, 3)
    disp_norms = np.linalg.norm(disp, axis=1)
    scale = max_displacement / disp_norms.max()

    phases = np.sin(2 * np.pi * np.arange(n_frames) / n_frames)
    all_frames = coords[None, :, :] + scale * phases[:, None, None] * disp[None, :, :]
    pad = 2.0

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=views[0], azim=views[1])
    ax.set_xlim(all_frames[..., 0].min() - pad, all_frames[..., 0].max() + pad)
    ax.set_ylim(all_frames[..., 1].min() - pad, all_frames[..., 1].max() + pad)
    ax.set_zlim(all_frames[..., 2].min() - pad, all_frames[..., 2].max() + pad)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    line, = ax.plot([], [], [], color='gray', lw=1)
    scatter = ax.scatter([], [], [], c=[], cmap=cmap, vmin=0, vmax=disp_norms.max())
    matplotlib.pyplot.colorbar(scatter, ax=ax, label='Mode displacement magnitude')

    def update(frame):
        pts = all_frames[frame]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        scatter.set_array(disp_norms)
        return line, scatter

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    matplotlib.pyplot.close(fig)