# note: documentation not written yet

import numpy as np

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets

from .consumption import ConsumerClass
from .firm import FirmClass1D


class KoopmanModel:
    def __init__(self, consumer, firm, L=24, e=0, w=1, **kwargs):

        # a. baseline setup
        self.consumer = ConsumerClass(**consumer)
        self.firm = FirmClass1D(**firm)

        # endowment
        self.L = 24
        self.e = 0

        # prices
        self.w = w

        # figure
        self.lmax = L
        if "xmax" not in kwargs:
            self.xmax = self.firm.g(L, self.firm.A, self.firm.gamma) * 1.5

        self.x1label = "$\\ell,L-f$"
        self.x2label = "$y,x$"

        self.N = 100

        # b. update
        for key, val in kwargs.items():
            setattr(self, key, val)

    ##########
    # figure #
    ##########

    def walras_figure(self):

        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        # a. set prices
        self.firm.w = self.w
        self.firm.p = 1
        self.consumer.p1 = self.w
        self.consumer.p2 = 1

        # b. firm
        self.firm.maximize_profit()
        self.firm.plot_max(ax, color=colors[0], label=f"firm choice")
        self.firm.plot_pmo_line(ax, color=colors[0], lmax=self.lmax, label="$g(\\ell)$")
        self.firm.plot_profit_line(
            ax, lmax=self.lmax, color="black", label="budgetline: $(\\pi-w\\ell)/p$"
        )

        # c. consumer
        self.consumer.I = self.w * self.L + 1 * self.e + self.firm.pi_ast
        self.consumer.maximize_utility()
        indiff_f, indiff_x = self.consumer.find_indifference_curve(
            u0=self.consumer.u_ast
        )

        # convert
        indiff_l = self.L - indiff_f
        f_ast = self.L - self.consumer.x1_ast
        x_ast = self.consumer.x2_ast - self.e

        # plot
        ax.plot(
            f_ast,
            x_ast,
            ls="",
            marker="*",
            markersize=7,
            color=colors[1],
            label=f"consumer choice",
        )
        ax.plot(
            indiff_l,
            indiff_x,
            color=colors[1],
            label=f"$u(L-\\ell,x) = {self.consumer.u_ast:.2f}$",
        )

        # d. layout
        ax.set_xlim([0, self.lmax])
        ax.set_ylim([0, self.xmax])
        ax.set_xlabel(self.x1label)
        ax.set_ylabel(self.x2label)

        ax.legend(loc="lower right")
        fig.tight_layout()


###############
# interactive #
###############


def _interactive_walras(alpha, beta, gamma, A, w, par):

    # a. consumer
    consumer = {}
    consumer["preferences"] = "cobb_douglas"
    consumer["alpha"] = alpha
    consumer["beta"] = beta

    # b. firm
    firm = {}
    firm["gamma"] = gamma
    firm["A"] = A

    # c. model
    model = KoopmanModel(consumer, firm, w=w, **par)
    model.walras_figure()


def interactive_walras(**kwargs):

    # a. preferences
    kwargs.setdefault("preferences", "cobb_douglas")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("alpha_min", 0.01)
    kwargs.setdefault("alpha_max", 5)
    kwargs.setdefault("alpha_step", 0.01)
    kwargs.setdefault("beta", 0.5)
    kwargs.setdefault("beta_min", 0.01)
    kwargs.setdefault("beta_max", 5)
    kwargs.setdefault("beta_step", 0.01)

    # b. production
    kwargs.setdefault("gamma", 0.1)
    kwargs.setdefault("gamma_min", 0.5)
    kwargs.setdefault("gamma_max", 0.9)
    kwargs.setdefault("gamma_step", 0.01)
    kwargs.setdefault("A", 0.5)
    kwargs.setdefault("A_min", 0.1)
    kwargs.setdefault("A_max", 10)
    kwargs.setdefault("A_step", 0.1)

    # c. figure
    widgets.interact(
        _interactive_walras,
        alpha=widgets.FloatSlider(
            description="$\\alpha$",
            min=kwargs["alpha_min"],
            max=kwargs["alpha_max"],
            step=kwargs["alpha_step"],
            value=kwargs["alpha"],
        ),
        beta=widgets.FloatSlider(
            description="$\\beta$",
            min=kwargs["beta_min"],
            max=kwargs["beta_max"],
            step=kwargs["beta_step"],
            value=kwargs["beta"],
        ),
        gamma=widgets.FloatSlider(
            description="$\\gamma$",
            min=kwargs["gamma_min"],
            max=kwargs["gamma_max"],
            step=kwargs["gamma_step"],
            value=kwargs["gamma"],
        ),
        A=widgets.FloatSlider(
            description="A",
            min=kwargs["A_min"],
            max=kwargs["A_max"],
            step=kwargs["A_step"],
            value=kwargs["A"],
        ),
        w=widgets.BoundedFloatText(
            description="$w$", min=0.05, max=2.00, step=0.001, value=0.2
        ),
        par=widgets.fixed(kwargs),
    )
