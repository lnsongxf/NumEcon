# note: documentation not written yet

import numpy as np
from types import SimpleNamespace

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets

#############
# functions #
#############


def f(l, k, par):
    return par.alpha / (1 + np.exp(-(l - par.beta)) + np.exp(-(k - par.gamma)))


def fmin(kbar, par):
    return par.alpha / (1 + np.exp(-(0 - par.beta)) + np.exp(-(kbar - par.gamma)))


def fmax(kbar, par):
    return par.alpha / (1 + np.exp(-(kbar - par.gamma)))


def C(x, w, r, par):
    Cw = w * (par.beta - np.log(w * (par.alpha - x) / (x * (w + r))))
    Cr = r * (par.gamma - np.log(r * (par.alpha - x) / (x * (w + r))))
    return Cw + Cr


def MC(x, w, r, par):
    return par.alpha * (w + r) / (x * (par.alpha - x))


def C_SR(x, w, kbar, par):
    kbar_fac = 1 + np.exp(-(kbar - par.gamma))
    return w * par.beta - w * np.log((par.alpha - kbar_fac * x) / x)


def MC_SR(x, w, kbar, par):
    kbar_fac = 1 + np.exp(-(kbar - par.gamma))
    denom = (par.alpha - kbar_fac * x) * x
    return w * par.alpha / denom


def supply(p, w, r, par):
    b = w + r
    nom = np.sqrt(par.alpha) * np.sqrt(par.alpha * p - 4 * b)
    denom = 2 * np.sqrt(p)
    return nom / denom + par.alpha / 2


def supply_SR(p, w, r, kbar, par):
    b = 1 + np.exp(-(kbar - par.gamma))
    nom = (
        np.sqrt(par.alpha) * np.sqrt(par.alpha * p - 4 * b * w) / np.sqrt(p) + par.alpha
    )
    denom = 2 * b
    return nom / denom


def pmin(w, r, par):
    b = w + r
    return 4 * b / par.alpha


def pmin_SR(w, r, kbar, par):
    b = 1 + np.exp(-(kbar - par.gamma))
    return 4 * b * w / par.alpha


#########
# total #
#########


def _cost_figure(par, w, r, kbar):

    ymax = 500

    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    x = np.linspace(1e-8, par.alpha - 1e-8, 1000)
    y = C(x, w, r, par)
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, color=colors[1], lw=2, label="$C^{LR}$")

    x = np.linspace(fmin(kbar, par) + 1e-8, fmax(kbar, par) - 1e-8, 1000)
    y = C_SR(x, w, kbar, par)
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, ls="--", color=colors[0], lw=2, label="$C_{\\bar{k}}$")
    ax.plot(x, y + r * kbar, ls="-", color=colors[0], lw=2, label="$TE_{\\bar{k}}$")

    # layout
    ax.set_xlim([0, par.alpha])
    ax.set_ylim([0, ymax])
    ax.grid(ls="--", lw=1)
    legend = ax.legend(loc="upper left", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")


def cost_figure():

    par = SimpleNamespace()
    par.alpha = 100
    par.beta = 5
    par.gamma = 5

    widgets.interact(
        _cost_figure,
        par=widgets.fixed(par),
        w=widgets.fixed(20),
        r=widgets.fixed(20),
        kbar=widgets.FloatSlider(
            description="$\\bar{k}$", min=1, max=20, step=0.1, value=8
        ),
    )


###########
# average #
###########


def _avg_cost_figure(par, w, r, kbar):

    ymax = 20

    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    x = np.linspace(1e-8, par.alpha - 1e-8, 1000)
    y = C(x, w, r, par) / x
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, color=colors[1], lw=2, label="$AC^{LR}$")

    x = np.linspace(fmin(kbar, par) + 1e-8, fmax(kbar, par) - 1e-8, 1000)
    y = C_SR(x, w, kbar, par) / x
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, ls="--", color=colors[0], lw=2, label="$AC_{\\bar{k}}$")

    y = (C_SR(x, w, kbar, par) + r * kbar) / x
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, ls="-", color=colors[0], lw=2, label="$AE_{\\bar{k}}$")

    # layout
    ax.set_xlim([0, par.alpha])
    ax.set_ylim([0, ymax])
    ax.grid(ls="--", lw=1)
    legend = ax.legend(loc="upper left", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")


def avg_cost_figure():

    par = SimpleNamespace()
    par.alpha = 100
    par.beta = 5
    par.gamma = 5

    widgets.interact(
        _avg_cost_figure,
        par=widgets.fixed(par),
        w=widgets.fixed(20),
        r=widgets.fixed(20),
        kbar=widgets.FloatSlider(
            description="$\\bar{k}$", min=1, max=20, step=0.1, value=8
        ),
    )


#############
# marginal  #
#############


def _MC_figure(par, w, r, kbar):

    ymax = 20

    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    x = np.linspace(1e-8, par.alpha - 1e-8, 1000)
    y = MC(x, w, r, par)
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, color=colors[1], lw=2, label="long-run")

    x = np.linspace(fmin(kbar, par) + 1e-8, fmax(kbar, par) - 1e-8, 1000)
    y = MC_SR(x, w, kbar, par)
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, color=colors[0], lw=2, label="short-run")

    # layout
    ax.set_xlim([0, par.alpha])
    ax.set_ylim([0, ymax])
    ax.grid(ls="--", lw=1)
    legend = ax.legend(loc="upper left", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")


def MC_figure():

    par = SimpleNamespace()
    par.alpha = 100
    par.beta = 5
    par.gamma = 5

    widgets.interact(
        _MC_figure,
        par=widgets.fixed(par),
        w=widgets.fixed(20),
        r=widgets.fixed(20),
        kbar=widgets.FloatSlider(
            description="$\\bar{k}$", min=1, max=20, step=0.1, value=8
        ),
    )


##########
# supply #
##########


def _supply_figure(par, pin, w, r, kbar):

    ymax = 20

    fig = plt.figure(figsize=(12, 4), dpi=100)

    # short run
    ax = fig.add_subplot(1, 2, 1)

    x = np.linspace(fmin(kbar, par) + 1e-8, fmax(kbar, par) - 1e-8, 1000)
    y = MC_SR(x, w, kbar, par)
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, ls="-", color=colors[1], lw=2, label="MC")

    x = np.linspace(fmin(kbar, par) + 1e-8, fmax(kbar, par) - 1e-8, 1000)
    y = C_SR(x, w, kbar, par) / x
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, ls="--", color=colors[0], lw=2, label="AC")
    ax.plot(x, y + r * kbar / x, ls="-", color=colors[0], lw=2, label="AE")

    p = np.linspace(pmin_SR(w, r, kbar, par) + 1e-8, 200, 1000)
    x = supply_SR(p, w, r, kbar, par)
    profit = p * x - C_SR(x, w, kbar, par)
    I = profit >= 0
    ax.plot(x[I], p[I], color="black", lw=6, alpha=0.50, label="supply", zorder=1)

    # text
    line = f""
    if pin >= pmin_SR(w, r, kbar, par):
        x = supply_SR(pin, w, r, kbar, par)
        if pin * x - C_SR(x, w, kbar, par) < 0:
            x = 0
            C_now = 0
        else:
            C_now = C_SR(x, w, kbar, par)
    else:
        x = 0
        C_now = 0
    ax.plot([0, x], [pin, pin], marker="o", color="black", lw=1)
    line += f"revenue: {np.round(pin*x,1)}\n"
    line += f"cost: {np.round(C_now,1)}\n"
    line += f"profit: {np.round(pin*x-C_now,1)}\n"
    line += f"fixed expenses: {np.round(r*kbar,1)}\n"
    ax.text(30, 14, line, fontsize=12)

    # layout
    ax.set_title("short-run")
    ax.set_xlim([0, par.alpha])
    ax.set_ylim([0, ymax])
    ax.grid(ls="--", lw=1)
    legend = ax.legend(loc="upper left", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")

    # long run
    ax = fig.add_subplot(1, 2, 2)

    x = np.linspace(1e-8, par.alpha - 1e-8, 1000)
    y = MC(x, w, r, par)
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, ls="-", color=colors[1], lw=2, label="MC")

    x = np.linspace(1e-8, par.alpha - 1e-8, 1000)
    y = C(x, w, r, par) / x
    x = np.append(x, x[-1])
    y = np.append(y, ymax)
    ax.plot(x, y, ls="--", color=colors[0], lw=2, label="AC")

    p = np.linspace(pmin(w, r, par) + 1e-8, 200, 1000)
    x = supply(p, w, r, par)
    I = x <= par.alpha
    p = p[I]
    x = x[I]
    profit = p * x - C(x, w, r, par)
    I = profit >= 0
    ax.plot(x[I], p[I], color="black", lw=6, alpha=0.50, label="supply", zorder=1)

    # text
    line = f""
    if pin >= pmin(w, r, par):
        x = supply(pin, w, r, par)
        if pin * x - C(x, w, r, par) < 0:
            x = 0
            C_now = 0
        else:
            C_now = C(x, w, kbar, par)
    else:
        x = 0
        C_now = 0
    ax.plot([0, x], [pin, pin], marker="o", color="black", lw=1)
    line += f"revenue: {np.round(pin*x,1)}\n"
    line += f"cost: {np.round(C_now,1)}\n"
    line += f"profit: {np.round(pin*x-C_now,1)}\n"
    ax.text(30, 15, line, fontsize=12)

    # layout
    ax.set_title("long-run")
    ax.set_xlim([0, par.alpha])
    ax.set_ylim([0, ymax])
    ax.grid(ls="--", lw=1)
    legend = ax.legend(loc="upper left", shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor("0.90")


def supply_figure():

    par = SimpleNamespace()
    par.alpha = 100
    par.beta = 7
    par.gamma = 7

    widgets.interact(
        _supply_figure,
        par=widgets.fixed(par),
        pin=widgets.FloatSlider(description="$p$", min=0, max=20, step=0.1, value=5),
        w=widgets.FloatSlider(description="$w$", min=0.1, max=40, step=1, value=20),
        r=widgets.FloatSlider(description="$r$", min=0.1, max=40, step=1, value=20),
        kbar=widgets.FloatSlider(
            description="$\\bar{k}$", min=0.1, max=20, step=0.1, value=9
        ),
    )
