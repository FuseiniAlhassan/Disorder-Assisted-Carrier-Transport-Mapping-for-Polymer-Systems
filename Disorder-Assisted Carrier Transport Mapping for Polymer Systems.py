import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Output directory
OUTDIR = '/mnt/data/disorder_transport_outputs'
os.makedirs(OUTDIR, exist_ok=True)

# --- Particle/Lattice Generation ---
def generate_particles(n, box, disorder=0.05, seed=None):
    rng = np.random.default_rng(seed)
    pos = rng.random((n,2)) * box
    pos += rng.normal(0, disorder, size=pos.shape)
    pos = np.clip(pos,0,box)
    return pos

# --- Network Construction ---
def compute_hopping_prob(dist, r0=0.02, p0=1.0):
    # anisotropic hopping probability
    return p0 * np.exp(-dist/r0)

def build_network(positions, cutoff=0.12):
    n = len(positions)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=positions[i])
    for i in range(n):
        for j in range(i+1,n):
            dist = np.linalg.norm(positions[i]-positions[j])
            if dist<=cutoff:
                G.add_edge(i,j,prob=compute_hopping_prob(dist))
    return G

# --- Local Conductance Mapping ---
def map_local_conductance(G, positions, box, grid_res=64):
    grid = np.zeros((grid_res, grid_res))
    xedges = np.linspace(0, box, grid_res+1)
    yedges = np.linspace(0, box, grid_res+1)
    for u,v,d in G.edges(data=True):
        mid = 0.5*(G.nodes[u]['pos'] + G.nodes[v]['pos'])
        ix = np.searchsorted(xedges, mid[0])-1
        iy = np.searchsorted(yedges, mid[1])-1
        ix = np.clip(ix,0,grid_res-1); iy = np.clip(iy,0,grid_res-1)
        grid[iy, ix] += d['prob']
    return grid

# --- Fourier Filtering ---
def fourier_lowpass(grid, frac=0.08):
    f = np.fft.fftshift(np.fft.fft2(grid))
    ny,nx = grid.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx,ky)
    K = np.sqrt(KX**2 + KY**2)
    cutoff = frac * 0.5
    mask = (K <= cutoff).astype(float)
    f_filtered = f*mask
    return np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

# --- Visualization ---
def plot_maps(grid, grid_filt, outdir):
    plt.figure(figsize=(5,4)); plt.imshow(grid, origin='lower'); plt.colorbar(); plt.title('Local Conductance Map'); plt.tight_layout(); plt.savefig(os.path.join(outdir,'conductance_map.png'),dpi=200); plt.close()
    plt.figure(figsize=(5,4)); plt.imshow(grid_filt, origin='lower'); plt.colorbar(); plt.title('Fourier Low-pass'); plt.tight_layout(); plt.savefig(os.path.join(outdir,'conductance_lowpass.png'),dpi=200); plt.close()

# --- Animation ---
def animate_transport(positions, box, outdir, steps=30):
    fig, ax = plt.subplots(figsize=(5,5))
    sc = ax.scatter(positions[:,0], positions[:,1], s=10, c='royalblue')
    ax.set_xlim(0,box); ax.set_ylim(0,box); ax.set_title('Carrier Transport Dynamics')
    def update(frame):
        steps_arr = np.random.normal(0,0.01,size=positions.shape)
        positions[:] = np.clip(positions+steps_arr,0,box)
        sc.set_offsets(positions)
        return sc,
    ani_path = os.path.join(outdir,'carrier_transport_animation.gif')
    ani = FuncAnimation(fig, update, frames=steps, interval=100)
    ani.save(ani_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return ani_path

# --- Run Example ---
n_particles = 300; box=1.0; seed=123
positions = generate_particles(n_particles, box, disorder=0.06, seed=seed)
G = build_network(positions, cutoff=0.12)
grid = map_local_conductance(G, positions, box, grid_res=64)
grid_filt = fourier_lowpass(grid, frac=0.06)
plot_maps(grid, grid_filt, OUTDIR)
ani_path = animate_transport(positions, box, OUTDIR)
display(Image(filename=ani_path))
print('All outputs saved to', OUTDIR)
