import jax.numpy as np
import numpy as npo
from jax.ops import index_add, index, index_update
import jax
from jax import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax.experimental import optimizers
from jax import value_and_grad
import argparse
import time
import matplotlib.animation as animation
from numpy import genfromtxt
import csv
from functools import partial
import pdb
import os

PI = np.pi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--keep_frac", type=float, default=0.5, help="fraction of phi we store"
)
parser.add_argument(
    "--filename",
    type=str,
    default="baseline",
    help="prefix for the .mp4 files and .csv file.",
)
parser.add_argument(
    "--num_opt",
    type=int,
    default=800,
    help="number of optimization iterations",
)

assert os.path.isdir("diff_data")

args = parser.parse_args()
anim_init_filename = "diff_data/{}_initial.mp4".format(args.filename)
anim_final_filename = "diff_data/{}_final.mp4".format(args.filename)
csvfile = "diff_data/{}_loss.csv".format(args.filename)

keep_frac = args.keep_frac

assert keep_frac <= 1.0
assert keep_frac > 0.0

# Need 4 D delta_t / (delta_x)^2 < 1 for numerical stability

# Here we set 4 D delta_t / (delta_x)^2 = 0.25

# Simulate for 10 units of time
T_f = 10
# 2D box with size 1 x 1
nx = 32
dx = 1 / nx
# Small timestep for numerical stability
dt = 1.0 / 4096
nt = int(T_f * 4096)
# 1/4 cancels factor of 4 in mean-squared distance
D = 0.25

N_optim = args.num_opt

rows, cols = nx + 1, nx + 1

x = np.linspace(0, nx * dx, nx + 1)
xg, yg = np.meshgrid(x, x)


sin2pix = np.sin(2 * PI * xg)
cos2pix = np.cos(2 * PI * xg)


def get_source(source_params, t):
    return (
        source_params[0]
        + source_params[1] * np.sin(PI * t)
        + source_params[2] * np.cos(PI * t)
        + source_params[3] * sin2pix * np.sin(PI * t)
        + source_params[4] * sin2pix * np.cos(PI * t)
        + source_params[5] * cos2pix * np.sin(PI * t)
        + source_params[6] * cos2pix * np.cos(PI * t)
    )


def get_ic():
    return np.sin(PI * xg) * np.sin(PI * yg)


ic = get_ic()


def get_perturbation():
    return 0.25 * np.sin(2 * PI * xg) * np.sin(PI * yg)


perturbation = get_perturbation()


def get_target(t):
    return ic + perturbation * np.sin(PI * t)


def step_phi(phi, S):
    """

    takes phi and returns phi at t+dt, using the update formula

    phi^{t+1}_{i,j} =  phi^t_{i,j} +
      (D dt / dx^2) * (phi^t_{i+1,j} + phi^t_{i,j+1} - 4 phi^t_{i,j} +
      phi^t_{i-1,j} + phi^t_{i,j-1})
    + dt * S^t_{ij} * phi^i_{ij}

    it all needs to be in the same index_add step otherwise we're not updating phi
    using the previous timestep

    """
    phi = index_add(
        phi,
        index[1:-1, 1:-1],
        (D * dt / dx ** 2)
        * (
            np.roll(phi, 1, axis=0)
            + np.roll(phi, 1, axis=1)
            + np.roll(phi, -1, axis=0)
            + np.roll(phi, -1, axis=1)
            - 4.0 * phi
        )[1:-1, 1:-1]
        + dt * S[1:-1, 1:-1] * phi[1:-1, 1:-1],
    )
    return phi


def loss_fn(phi, target):
    return np.mean((phi - target) ** 2)


def sim_step(phi, source_params, t):
    source = get_source(source_params, t)
    target = get_target(t)
    phi = step_phi(phi, source)
    dl = loss_fn(phi, target) * dt
    return phi, dl


def sim_step_target_sampled(phi, source_params, t, target):
    source = get_source(source_params, t)
    phi = step_phi(phi, source)
    dl = loss_fn(phi, target) * dt
    return phi, dl


@jax.custom_vjp
def sim_step_subsampled(phi, source_params, t, key):
    phi, dl = sim_step(phi, source_params, t)
    return phi, dl


def sim_step_sub_fwd(phi, source_params, t, key):
    n_ixs = rows * cols
    all_ixs = np.arange(n_ixs, dtype=np.int32)
    keep_ixs = jax.random.shuffle(key, all_ixs)[: int(n_ixs * keep_frac)]

    phi_sub = phi.ravel()[keep_ixs]

    primals = sim_step_subsampled(phi, source_params, t, key)

    return primals, (phi_sub, source_params, t, key)


def sim_step_sub_rev(res, g):

    phi_sub, source_params, t, key = res
    n_ixs = rows * cols
    all_ixs = np.arange(n_ixs, dtype=np.int32)
    keep_ixs = jax.random.shuffle(key, all_ixs)[: int(n_ixs * keep_frac)]

    phi = np.zeros((rows, cols)).ravel()
    phi = jax.ops.index_add(phi, keep_ixs, phi_sub)
    phi = phi.reshape(rows, cols)
    new_target = np.zeros((rows, cols)).ravel()
    new_target = jax.ops.index_add(
        new_target, keep_ixs, get_target(t).ravel()[keep_ixs]
    )
    new_target = new_target.reshape(rows, cols)

    def forward(phi, source_params):
        return sim_step_target_sampled(phi, source_params, t, new_target)

    primals, vjp = jax.vjp(forward, phi, source_params)
    grad_phi, grad_source = vjp(g)
    return (grad_phi, grad_source, np.zeros_like(t), np.zeros_like(key))


sim_step_subsampled.defvjp(sim_step_sub_fwd, sim_step_sub_rev)


def simulate(source_params, ic, key=jax.random.PRNGKey(0)):
    scan_step = lambda phi, t_and_key: sim_step_subsampled(
        phi, source_params, t_and_key[0], t_and_key[1]
    )
    scan_inputs = (np.linspace(0, T_f, nt + 1), jax.random.split(key, nt + 1))

    phi, losses = jax.lax.scan(scan_step, ic, scan_inputs)
    return np.sum(losses)


def print_loss(loss, step):
    print("step is : {} loss is: {}".format(step, loss))


def write_csv(losses, csvfile):
    with open(csvfile, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(losses)


def animate(source_params, ic, filename):
    scan_step = lambda phi, t: anim_step(phi, source_params, t)
    _, phis = jax.lax.scan(scan_step, ic, np.linspace(0, T_f, nt + 1))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax = ax1, ax2
    ax1.set_zlim(0, 1)
    ax2.set_zlim(0, 1)
    ani = animation.FuncAnimation(
        fig,
        lambda i: anim_callback(int(i), phis, ax),
        blit=False,
        frames=np.arange(0, 40000, 1000),
    )
    ani.save(filename)


def anim_callback(i, phis, ax):
    ax1, ax2 = ax
    t = i * dt
    target = get_target(t)
    ax1.clear()
    ax2.clear()
    ax1.set_zlim(0, 1)
    ax2.set_zlim(0, 1)
    ax1.set_title("Neutron Density")
    ax2.set_title("Target Neutron Density")
    surface = ax1.plot_surface(xg, yg, phis[i, :, :])
    surface = ax2.plot_surface(xg, yg, target)
    return surface


def anim_step(phi, source_params, t):
    source = get_source(source_params, t)
    phi = step_phi(phi, source)
    return phi, phi


key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
source_params = jax.random.uniform(key, minval=-0.1, maxval=0.1, shape=(7,))
source_params = index_update(
    source_params, index[0], 4.932
)  # 4.932 will give steady state
ic = get_ic()

objective_with_grad = jit(
    value_and_grad(lambda source_params: simulate(source_params, ic))
)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)

opt_state = opt_init(source_params)

animate(source_params, ic, anim_init_filename)

t1 = time.time()
losses = npo.zeros(N_optim)
for step in range(N_optim):
    l, g = objective_with_grad(get_params(opt_state))
    opt_state = opt_update(step, g, opt_state)
    print_loss(l, step)
    losses[step] = l

write_csv(losses, csvfile)

t2 = time.time()
print("Time to run: {}".format(t2 - t1))

animate(get_params(opt_state), ic, anim_final_filename)
