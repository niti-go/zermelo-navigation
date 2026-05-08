"""
2D Taylor-Green Vortex — Analytical Solution Visualization
===========================================================
Plots velocity field (quiver), vorticity, and energy decay.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ──────────────────────────────────────────────────────────────
U0  = 1.0       # Velocity amplitude
nu  = 0.01      # Kinematic viscosity
rho = 1.0       # Density
N   = 32        # Grid resolution

# ── Grid ────────────────────────────────────────────────────────────────────
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
y = np.linspace(0, 2 * np.pi, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# ── Analytical solution ──────────────────────────────────────────────────────
def taylor_green(X, Y, t, U0=1.0, nu=0.01, rho=1.0):
    decay_v = np.exp(-2 * nu * t)
    decay_p = np.exp(-4 * nu * t)
    u = U0 * np.sin(X) * np.cos(Y) * decay_v
    v = -U0 * np.cos(X) * np.sin(Y) * decay_v
    p = (rho * U0**2 / 4) * (np.cos(2*X) + np.cos(2*Y)) * decay_p
    omega = -2 * U0 * np.sin(X) * np.sin(Y) * decay_v      # z-vorticity
    speed = np.sqrt(u**2 + v**2)
    return u, v, p, omega, speed

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor('#0f1117')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

BG   = '#0f1117'
TEXT = '#e0e0e0'
cmap_v = 'RdBu_r'      # vorticity
cmap_s = 'inferno'     # speed / pressure

times = [0.0, 5.0, 20.0]
titles = ['$t = 0$', '$t = 5$', '$t = 20$']

# ── Top row: velocity + vorticity at three times ─────────────────────────────
for col, (t, title) in enumerate(zip(times, titles)):
    ax = fig.add_subplot(gs[0, col])
    u, v, p, omega, speed = taylor_green(X, Y, t, U0, nu, rho)

    # Vorticity background
    im = ax.contourf(X, Y, omega, levels=60, cmap=cmap_v,
                     vmin=-2*U0, vmax=2*U0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('$\\omega_z$', color=TEXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

    # Velocity arrows (skip every other point for clarity)
    skip = 2
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              speed[::skip, ::skip], cmap='autumn',
              scale=12, width=0.006, alpha=0.9)

    ax.set_facecolor(BG)
    ax.set_title(f'Velocity & vorticity  {title}', color=TEXT, fontsize=11, pad=8)
    ax.set_xlabel('$x$', color=TEXT, fontsize=10)
    ax.set_ylabel('$y$', color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(['0', '$\\pi$', '$2\\pi$'], color=TEXT)
    ax.set_yticks([0, np.pi, 2*np.pi])
    ax.set_yticklabels(['0', '$\\pi$', '$2\\pi$'], color=TEXT)

# ── Bottom-left: Energy decay ─────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 0])
t_arr = np.linspace(0, 50, 500)
E0 = U0**2 / 4
E  = E0 * np.exp(-4 * nu * t_arr)
ax_e.plot(t_arr, E,  color='#00cfff', lw=2, label='$E(t)$')
ax_e.plot(t_arr, nu * U0**2 * np.exp(-4 * nu * t_arr),
          color='#ff6b6b', lw=2, linestyle='--', label='$\\varepsilon(t)$')
ax_e.set_facecolor(BG)
ax_e.set_xlabel('Time $t$', color=TEXT)
ax_e.set_ylabel('Amplitude', color=TEXT)
ax_e.set_title('Kinetic energy & dissipation', color=TEXT, fontsize=11)
ax_e.legend(facecolor='#1e1e2e', edgecolor='#444', labelcolor=TEXT)
ax_e.tick_params(colors=TEXT)
for sp in ax_e.spines.values(): sp.set_edgecolor('#444')

# ── Bottom-centre: Pressure field at t=0 ────────────────────────────────────
ax_p = fig.add_subplot(gs[1, 1])
_, _, p0, _, _ = taylor_green(X, Y, 0, U0, nu, rho)
im2 = ax_p.contourf(X, Y, p0, levels=60, cmap='plasma')
cb2 = fig.colorbar(im2, ax=ax_p, fraction=0.046, pad=0.04)
cb2.set_label('$p$', color=TEXT, fontsize=9)
cb2.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=TEXT)
ax_p.set_facecolor(BG)
ax_p.set_title('Pressure field  $t = 0$', color=TEXT, fontsize=11, pad=8)
ax_p.set_xlabel('$x$', color=TEXT); ax_p.set_ylabel('$y$', color=TEXT)
ax_p.tick_params(colors=TEXT)
for sp in ax_p.spines.values(): sp.set_edgecolor('#444')
ax_p.set_xticks([0, np.pi, 2*np.pi])
ax_p.set_xticklabels(['0', '$\\pi$', '$2\\pi$'], color=TEXT)
ax_p.set_yticks([0, np.pi, 2*np.pi])
ax_p.set_yticklabels(['0', '$\\pi$', '$2\\pi$'], color=TEXT)

# ── Bottom-right: Streamlines at t=0 ────────────────────────────────────────
ax_s = fig.add_subplot(gs[1, 2])
u0, v0, _, _, speed0 = taylor_green(X, Y, 0, U0, nu, rho)
x1d = np.linspace(0, 2*np.pi, N)
ax_s.streamplot(x1d, x1d, u0, v0,
                color=speed0, cmap='cool',
                linewidth=1.2, density=1.4, arrowsize=1.2)
ax_s.set_facecolor(BG)
ax_s.set_title('Streamlines  $t = 0$', color=TEXT, fontsize=11, pad=8)
ax_s.set_xlabel('$x$', color=TEXT); ax_s.set_ylabel('$y$', color=TEXT)
ax_s.tick_params(colors=TEXT)
for sp in ax_s.spines.values(): sp.set_edgecolor('#444')
ax_s.set_xticks([0, np.pi, 2*np.pi])
ax_s.set_xticklabels(['0', '$\\pi$', '$2\\pi$'], color=TEXT)
ax_s.set_yticks([0, np.pi, 2*np.pi])
ax_s.set_yticklabels(['0', '$\\pi$', '$2\\pi$'], color=TEXT)
ax_s.set_xlim(0, 2*np.pi); ax_s.set_ylim(0, 2*np.pi)

# ── Super-title ──────────────────────────────────────────────────────────────
fig.suptitle('2D Taylor–Green Vortex  ·  Analytical Solution\n'
             r'$u = U_0\sin x\cos y\,e^{-2\nu t}$   '
             r'$v = -U_0\cos x\sin y\,e^{-2\nu t}$   '
             r'$(\nu=' + str(nu) + r',\ U_0=' + str(U0) + r')$',
             color=TEXT, fontsize=13, y=0.98)

plt.savefig('taylor_green_vortex.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved → taylor_green_vortex.png")
