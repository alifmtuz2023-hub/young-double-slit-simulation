import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.ndimage import uniform_filter

# ================================================================
#  Young's Double-Slit Experiment  -- LIVE ANIMATION
#  تجربة يونج للشقين  -- محاكاة حية بالحركة الكاملة
#  Finite-difference wave propagation (real-time)
# ================================================================

# ------- parameters (editable via sliders) -------
N        = 300        # grid resolution
boxsize  = 1.0
c        = 1.0        # wave speed
dt       = (np.sqrt(2)/2) * (boxsize/N) / c
dx       = boxsize / N
fac      = dt**2 * c**2 / dx**2

# slit parameters
slit_pos    = int(N * 9/32)   # barrier x-position
slit_half   = int(N * 3/64)   # half-width of each slit opening
slit_sep    = int(N * 5/32)   # distance between slit centers (half)
freq        = 20.0             # source frequency

xlin = np.linspace(0.5*dx, boxsize - 0.5*dx, N)

def build_mask(n, sp, sh, ss):
    mask = np.zeros((n, n), dtype=bool)
    mask[0, :]  = True
    mask[-1, :] = True
    mask[:, 0]  = True
    mask[:, -1] = True
    # barrier row band
    mask[sp : sp+int(n/16), :] = True
    # carve slit 1
    c1 = n//2 + ss
    mask[sp : sp+int(n/16), max(0,c1-sh):c1+sh] = False
    # carve slit 2
    c2 = n//2 - ss
    mask[sp : sp+int(n/16), max(0,c2-sh):c2+sh] = False
    return mask

mask  = build_mask(N, slit_pos, slit_half, slit_sep)
U     = np.zeros((N, N))
Uprev = np.zeros((N, N))
t_sim = [0.0]

# ------- figure layout -------
plt.rcParams['figure.facecolor'] = '#0a0a14'
plt.rcParams['axes.facecolor']   = '#0a0a14'

fig = plt.figure(figsize=(14, 7))
gs  = gridspec.GridSpec(2, 3,
    left=0.06, right=0.97, top=0.92, bottom=0.15,
    hspace=0.45, wspace=0.30,
    height_ratios=[1, 0.12],
    width_ratios=[2.5, 1.0, 0.9])

ax_wave  = fig.add_subplot(gs[0, 0])   # 2-D wave field
ax_inten = fig.add_subplot(gs[0, 1])   # screen intensity
ax_info  = fig.add_subplot(gs[0, 2])   # info box
ax_sfreq = fig.add_subplot(gs[1, 0])   # slider: frequency
ax_ssep  = fig.add_subplot(gs[1, 1])   # slider: slit sep

for ax in [ax_wave, ax_inten]:
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#555')

cmap = plt.cm.seismic      # blue=negative, red=positive
cmap.set_bad('#2a2a2a')    # barrier = dark gray

Uplot = np.where(mask, np.nan, U)
im = ax_wave.imshow(
    Uplot.T, cmap=cmap, origin='lower',
    extent=[0,1,0,1], aspect='equal',
    vmin=-2, vmax=2, animated=True)

ax_wave.set_title("Wave Field  (live)\nتجربة يونج - محاكاة حية",
                  color='white', fontsize=11, fontweight='bold')
ax_wave.set_xlabel('x', color='white'); ax_wave.set_ylabel('y', color='white')
cb = fig.colorbar(im, ax=ax_wave, fraction=0.03, pad=0.02)
cb.set_label('Amplitude', color='white', fontsize=8)
cb.ax.tick_params(colors='white')

# screen intensity line
ys   = np.linspace(0, 1, N)
I0   = np.zeros(N)
line_I, = ax_inten.plot(I0, ys, color='#FFD700', lw=1.8)
fill_I  = ax_inten.fill_betweenx(ys, 0, I0, alpha=0.3, color='#FFD700')
ax_inten.set_xlim(0, 5)
ax_inten.set_ylim(0, 1)
ax_inten.set_xlabel('Intensity', color='white', fontsize=9)
ax_inten.set_ylabel('y', color='white', fontsize=9)
ax_inten.set_title('Screen\nIntensity', color='white', fontsize=10, fontweight='bold')
ax_inten.grid(True, color='#222', lw=0.5)

# info text
ax_info.axis('off')
ax_info.set_facecolor('#0a0a14')
info_txt = ax_info.text(
    0.05, 0.5,
    '',
    color='#00FF99', fontsize=9,
    family='monospace', va='center',
    transform=ax_info.transAxes,
    bbox=dict(boxstyle='round,pad=0.5', fc='#0d1a0d', ec='#00FF99', lw=1))

# sliders
slider_freq = Slider(ax_sfreq, 'Frequency', 5, 60,
                     valinit=freq, color='#3366ff')
slider_sep  = Slider(ax_ssep,  'Slit sep', 5, int(N*0.25),
                     valinit=slit_sep, valstep=1, color='#ff6633')
for sl in [slider_freq, slider_sep]:
    sl.label.set_color('white')
    sl.valtext.set_color('white')

# title
fig.suptitle(
    "Young's Double-Slit  |  تجربة يونج للشقين  |  LIVE Simulation",
    color='white', fontsize=13, fontweight='bold', y=0.97)

# ------- update on slider change -------
def reset_sim(*args):
    global mask, U, Uprev
    freq_new = slider_freq.val
    sep_new  = int(slider_sep.val)
    mask  = build_mask(N, slit_pos, slit_half, sep_new)
    U     = np.zeros((N, N))
    Uprev = np.zeros((N, N))
    t_sim[0] = 0.0

slider_freq.on_changed(reset_sim)
slider_sep.on_changed(reset_sim)

# ------- animation frame -------
STEPS_PER_FRAME = 8      # physics steps per rendered frame

def update(frame):
    global U, Uprev
    fr = slider_freq.val
    for _ in range(STEPS_PER_FRAME):
        ULX = np.roll(U,  1, axis=0)
        URX = np.roll(U, -1, axis=0)
        ULY = np.roll(U,  1, axis=1)
        URY = np.roll(U, -1, axis=1)
        lap = ULX + URX + ULY + URY - 4*U
        Unew = 2*U - Uprev + fac*lap
        Uprev[:] = U
        U[:] = Unew
        U[mask] = 0
        # source: oscillating wave from left edge
        U[0, :] = np.sin(2*np.pi*fr*t_sim[0]) * np.sin(np.pi*xlin)**2
        t_sim[0] += dt

    Uplot = np.where(mask, np.nan, U)
    im.set_data(Uplot.T)

    # screen intensity: right column
    I_screen = U[-2, :]**2
    I_screen = uniform_filter(I_screen, size=5)
    line_I.set_xdata(I_screen)

    # rebuild fill
    global fill_I
    fill_I.remove()
    fill_I = ax_inten.fill_betweenx(
        ys, 0, I_screen, alpha=0.3, color='#FFD700')
    ax_inten.set_xlim(0, max(I_screen.max()*1.2, 0.1))

    # info box
    info_txt.set_text(
        f'  t = {t_sim[0]:.3f} s\n'
        f'  f = {fr:.0f} Hz\n'
        f'  N = {N}\n'
        f'  dt= {dt:.5f}\n'
        f'  steps/frame = {STEPS_PER_FRAME}')

    return im, line_I, info_txt

# ------- run -------
ani = FuncAnimation(
    fig, update,
    frames=None,      # run forever
    interval=30,      # ms between frames (~33 fps)
    blit=False)

plt.show()
