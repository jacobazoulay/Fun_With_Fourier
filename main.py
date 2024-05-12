import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pyautogui
import time

def get_mouse():
    pts = []
    for i in range(40):
        x, y = pyautogui.position()
        pts.append([x / 300, y / 300])
        time.sleep(0.2)
    print(pts)

def fheart(t):
    T = 2 * np.pi
    pts = [[0, 0], [0.1, 0.4], [0.2, 0.0], [0.17, -0.2]]
    pts = [[2.1133333333333333, 1.5833333333333333], [2.203333333333333, 1.4533333333333334], [2.24, 1.4], [2.3066666666666666, 1.32], [2.4, 1.23], [2.53, 1.1666666666666667], [2.6466666666666665, 1.1566666666666667], [2.75, 1.2366666666666666], [2.77, 1.4266666666666667], [2.6433333333333335, 1.6066666666666667], [2.48, 1.7266666666666666], [2.32, 1.8233333333333333], [2.1866666666666665, 1.93], [2.09, 2.0966666666666667], [2.0366666666666666, 2.243333333333333], [2.03, 2.276666666666667], [2.02, 2.2133333333333334], [1.9866666666666666, 2.1066666666666665], [1.9333333333333333, 2.02], [1.8766666666666667, 1.9633333333333334], [1.8, 1.9033333333333333], [1.7066666666666668, 1.8166666666666667], [1.6433333333333333, 1.74], [1.5966666666666667, 1.6266666666666667], [1.5933333333333333, 1.4633333333333334], [1.62, 1.3033333333333332], [1.6666666666666667, 1.17], [1.7333333333333334, 1.0266666666666666], [1.8066666666666666, 0.9666666666666667], [1.9166666666666667, 0.95], [2.02, 0.9566666666666667], [2.0966666666666667, 1.04], [2.14, 1.1666666666666667], [2.1466666666666665, 1.2866666666666666], [2.1466666666666665, 1.3766666666666667], [2.1466666666666665, 1.45], [2.1466666666666665, 1.49], [2.1466666666666665, 1.5266666666666666], [2.1466666666666665, 1.5266666666666666], [2.1466666666666665, 1.5266666666666666]]

    N = len(pts)
    t_step = T / N
    step_id = t / t_step
    id_start = int(step_id) % N
    id_end = (id_start + 1) % N

    w = step_id - id_start
    r = pts[id_start][0] * (1- w) + pts[id_end][0] * w
    i = pts[id_start][1] * (1 - w) + pts[id_end][1] * w
    return r + 1j * i


def fsquare(t):
    if t < np.pi / 2:
        r = -1 * t / (np.pi / 2) + 1
        i = 1
    elif t < np.pi:
        r = 0
        i = -1 * (t - np.pi / 2) / (np.pi / 2) + 1
    elif t < np.pi * (3 / 2):
        r = 1 * (t - np.pi) / (np.pi / 2)
        i = 0
    else:
        r = 1
        i = 1 * (t - (np.pi * (3 / 2))) / (np.pi / 2)
    return r + 1j * i


def f4(t):
    return np.cos(t) + np.sin(t) * 1j


def f3(t):
    return 1.5*np.cos(t) - np.cos(30*t) + (1.5*np.sin(t) - np.sin(30*t))*1j


def f2(t):
    t = t - 4
    return (4*np.sin(4*t)) + (3*np.sin(3*t)) * 1j


def f(t):
    if t < np.pi/2:
        out = -1
    elif t < np.pi:
        out = -0.5
    elif t < np.pi * (3/2):
        out = 0.5
    else:
        out = 1
    return out


def f_params(f, n, T):
    out = lambda t: f(t) * np.exp(-2 * np.pi * n * t * 1j / T)
    return out


def calc_nth_coef(f_t, n, T):
    real_integrand = lambda t: np.real(f_params(f_t, n, T)(t))
    imag_integrand = lambda t: np.imag(f_params(f_t, n, T)(t))

    c_n_real = (1 / T) * quad(real_integrand, 0, T)[0]
    c_n_imag = (1 / T) * quad(imag_integrand, 0, T)[0]

    c_n = c_n_real + (c_n_imag * 1j)

    c_n = round_complex(c_n, 6)

    return c_n


def calc_c_ns(f, N, T=2*np.pi):
    c_ns = []
    for i in range(-1 * N, N + 1):
        c_n = calc_nth_coef(f, i, np.pi * 2)
        c_ns.append(c_n)
    return c_ns


def round_complex(x, n):
    return complex(round(x.real, n), round(x.imag, n))


def compose_func_from_c_ns(t, c_ns, T):
    out = 0.0 + 0.0j
    for i in range(-1 * (len(c_ns) // 2), len(c_ns) // 2 + 1):
        y_n = c_ns[i + len(c_ns) // 2] * np.exp(2 * np.pi * i * t * 1j / T)
        out += y_n

    return out


def compose_func_bases_from_c_ns(t, c_ns, T):
    vecs = []
    for i in range(-1 * (len(c_ns) // 2), len(c_ns) // 2 + 1):
        y_n = c_ns[i + len(c_ns) // 2] * np.exp(2 * np.pi * i * t * 1j / T)
        vecs.append(y_n)

    return vecs


def get_ys(c_ns, T):
    ts = np.linspace(0, T, 5000)
    ys = []
    for t in ts:
        y = compose_func_from_c_ns(t, c_ns, T)
        ys.append(y)

    return ts, ys


def get_vecs(c_ns, T):
    ts = np.linspace(0, T, 5000)
    vecs = []
    for t in ts:
        bases = compose_func_bases_from_c_ns(t, c_ns, T)
        vecs.append(bases)

    return ts, vecs


def plot_fourier_sum(c_ns, T):
    ts, ys = get_ys(c_ns, T)

    plt.plot(ts, ys)
    plt.xlim(min(ts), max(ts))
    plt.ylim(min(ys), max(ys))
    plt.ylabel('f(t)')
    plt.xlabel('t')
    plt.show()


def get_real_imag_from_ys(ys):
    # extract real part
    xs = [ele.real for ele in ys]
    # extract imaginary part
    ys = [ele.imag for ele in ys]

    return xs, ys


def plot_complex(c_ns, T):
    ts, ys = get_ys(c_ns, T)
    x_real, y_imag = get_real_imag_from_ys(ys)

    # plot the complex numbers
    plt.xlim(min(x_real), max(x_real))
    plt.ylim(min(y_imag), max(y_imag))
    plt.plot(x_real, y_imag)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()


def reorder_vecs(vecs):
    new_vecs = [vecs[len(vecs)//2]]
    N = len(vecs) // 2
    for i in range(1, N + 1):
        pos_id = i + N
        neg_id = -i + N
        new_vecs.append(vecs[pos_id])
        new_vecs.append(vecs[neg_id])
    return new_vecs


def plot_circle(mid, r_vec, ax=None):
    x_c, y_c = np.real(mid), np.imag(mid)
    r = np.sqrt(np.real(r_vec)**2 + np.imag(r_vec)**2)

    xs = []
    ys = []
    for t in np.linspace(0, np.pi * 2, 50):
        x = x_c + r * np.cos(t)
        y = y_c + r * np.sin(t)

        xs.append(x)
        ys.append(y)

    if ax is None:
        plt.plot(xs, ys, color='gray', linestyle='dashed', linewidth=0.5)
    else:
        ax.plot(xs, ys, color='gray', linestyle='dashed', linewidth=0.5)


def plot_complex_with_vecs(c_ns, T):
    fig, ax = plt.subplots()
    ts, ys = get_ys(c_ns, T)
    x_reals, y_imags = get_real_imag_from_ys(ys)
    ax.set_aspect('equal')
    ax_lims = (np.min(x_reals + y_imags) * 1.1, np.max(x_reals + y_imags) * 1.1)

    for t in ts:
        plt.plot(x_reals, y_imags, linewidth=1)
        vecs = compose_func_bases_from_c_ns(t, c_ns, T)
        vecs = reorder_vecs(vecs)
        total = 0 + 0j
        for vec in vecs:
            if vec != 0.0 + 0.0j:
                xs = [np.real(x) for x in [total, total+vec]]
                ys = [np.imag(y) for y in [total, total+vec]]
                plt.plot(xs, ys, linewidth=1)
                plot_circle(total, vec)
                total += vec
        plt.xlim((-3, 3))
        plt.ylim((-3, 3))
        plt.scatter(np.real(total), np.imag(total), color='r', marker='o')
        plt.show()


def animate_complex_with_vecs(c_ns, T, zoomed_flag=False, zoomed_border=0.01, speed=1.0, fps=30):
    fig, ax = plt.subplots()
    ts, ys = get_ys(c_ns, T)
    x_reals, y_imags = get_real_imag_from_ys(ys)
    ax.set_aspect('equal')
    ax_lims = (np.min(x_reals + y_imags) - 1, np.max(x_reals + y_imags) + 1 )

    def update(frame):
        ax.clear()  # Clear the previous plot
        plt.title(f'Frame {frame + 1}')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        ax.plot(x_reals, y_imags, linewidth=1)

        total = 0 + 0j
        t = (T * frame) / num_frames
        vecs = compose_func_bases_from_c_ns(t, c_ns, T)
        vecs = reorder_vecs(vecs)

        for vec in vecs:
            if True: #np.real(vec) > 0.00001 and np.imag(vec) > 0.00001:
                xs = [np.real(x) for x in [total, total+vec]]
                ys = [np.imag(y) for y in [total, total+vec]]
                ax.plot(xs, ys, linewidth=1)
                plot_circle(total, vec, ax)
                total += vec

        ax.scatter(np.real(total), np.imag(total), color='r', marker='o')
        if zoomed_flag:
            ax.set_xlim((np.real(total)-zoomed_border, np.real(total) + zoomed_border))
            ax.set_ylim((np.imag(total)-zoomed_border, np.imag(total)+zoomed_border))
        else:
            ax.set_xlim(ax_lims)
            ax.set_ylim(ax_lims)

    num_frames = int(T * fps)
    ani = manimation.FuncAnimation(fig, update, frames=num_frames, repeat=True, interval=int(1000/(fps*speed)))
    plt.show()


def main():
    T = 2 * np.pi
    N = 20
    zoomed_flag = True
    zoomed_border = 0.05
    speed = 0.02
    fps = 500

    c_ns = calc_c_ns(fheart, N, T)
    # plot_fourier_sum(c_ns, T)
    # plot_complex(c_ns, T)
    # plot_complex_with_vecs(c_ns, T)

    # c_ns = [1j, -3j, 2j, 0.6j, 0j, -0.6j, -2j, 3j, -1j]
    # plot_complex(c_ns, T)
    # plot_complex_with_vecs(c_ns, T)
    # get_mouse()

    animate_complex_with_vecs(c_ns, T, zoomed_flag, zoomed_border, speed, fps)


if __name__ == '__main__':
    main()
