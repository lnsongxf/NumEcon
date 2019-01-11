# note: documentation not written yet

import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets


class ConsumerClass:
    def __init__(self, **kwargs):

        # a. baseline setup

        # utility
        self.preferences = "cobb_douglas"
        self.alpha = 1
        self.beta = 1
        self.indifference_curves_method = "analytical"
        self.utility_max_method = "analytical"
        self.cost_min_method = "analytical"
        self.monotone = True

        # prices and income
        self.budgetsettype = "exogenous"
        self.p1 = 1
        self.p2 = 1
        self.I = 10
        self.e1 = np.nan
        self.e2 = np.nan
        self.kink_point = np.nan
        self.kink_slope = np.nan

        # figures
        self.x1_label = "$x_1$"
        self.x2_label = "$x_2$"
        self.x1_scale = 1.2
        self.x2_scale = 1.2
        self.N = 100

        # figure limits
        if "x1max" in kwargs and "x2max" in kwargs:
            self.automatic_lims = False
        else:
            self.automatic_lims = True

        # b. update
        for key, val in kwargs.items():
            setattr(self, key, val)

        # c. calculations
        self.calculations()

    def update_income(self):

        self.I = self.p1 * self.e1 + self.p2 * self.e2

    def calculations(self):

        # a. income
        if self.budgetsettype == "endogenous":
            self.update_income()

        # b. utility function
        if self.preferences == "cobb_douglas":
            self.utility_func = lambda x1, x2, alpha, beta: x1 ** alpha * x2 ** beta
            self.indiff_func_x1 = lambda u0, x1, alpha, beta: (u0 / x1 ** alpha) ** (
                1 / beta
            )
            self.indiff_func_x2 = lambda u0, x2, alpha, beta: (u0 / x2 ** beta) ** (
                1 / alpha
            )
        elif self.preferences == "ces":
            self.utility_func = (
                lambda x1, x2, alpha, beta: (
                    alpha * x1 ** (-beta) + (1 - alpha) * x2 ** (-beta)
                )
                ** (-1 / beta)
                if beta != 0
                else x1 ** alpha * x2 ** (1 - alpha)
            )
            self.indiff_func_x1 = (
                lambda u0, x1, alpha, beta: (
                    (u0 ** (-beta) - alpha * x1 ** (-beta)) / (1 - alpha)
                )
                ** (-1 / beta)
                if beta != 0
                else (u0 / x1 ** alpha) ** (1 / (1 - alpha))
            )
            self.indiff_func_x2 = (
                lambda u0, x2, alpha, beta: (
                    (u0 ** (-beta) - (1 - alpha) * x2 ** (-beta)) / alpha
                )
                ** (-1 / beta)
                if beta != 0
                else (u0 / x2 * (1 - alpha)) ** (1 / alpha)
            )
            self.indifference_curves_method = "numerical"
            self.utility_max_method = "numerical"
            self.cost_min_method = "numerical"
        elif self.preferences == "perfect_substitutes":
            self.utility_func = lambda x1, x2, alpha, beta: alpha * x1 + beta * x2
            self.indiff_func_x1 = lambda u0, x1, alpha, beta: (u0 - alpha * x1) / beta
            self.indiff_func_x2 = lambda u0, x2, alpha, beta: (u0 - beta * x2) / alpha
            self.cost_min_method = "numerical"
        elif self.preferences == "perfect_complements":
            self.utility_func = lambda x1, x2, alpha, beta: np.fmin(
                alpha * x1, beta * x2
            )
            self.indifference_curves_method = self.preferences
            self.cost_min_method = "numerical"
        elif self.preferences == "quasi_log":
            self.utility_func = (
                lambda x1, x2, alpha, beta: alpha * np.log(x1) + x2 * beta
            )
            self.indiff_func_x1 = (
                lambda u0, x1, alpha, beta: (u0 - alpha * np.log(x1)) / beta
            )
            self.indiff_func_x2 = lambda u0, x2, alpha, beta: np.exp(
                (u0 - x2 * beta) / alpha
            )
        elif self.preferences == "quasi_sqrt":
            self.utility_func = (
                lambda x1, x2, alpha, beta: alpha * np.sqrt(x1) + x2 * beta
            )
            self.indiff_func_x1 = (
                lambda u0, x1, alpha, beta: (u0 - alpha * np.sqrt(x1)) / beta
            )
            self.indiff_func_x2 = (
                lambda u0, x2, alpha, beta: ((np.fmax(u0 - x2 * beta, 0)) / alpha) ** 2
            )
            self.cost_min_method = "numerical"
        elif self.preferences == "concave":
            self.utility_func = (
                lambda x1, x2, alpha, beta: alpha * x1 ** 2 + beta * x2 ** 2
            )
            self.indiff_func_x1 = lambda u0, x1, alpha, beta: np.sqrt(
                (u0 - alpha * x1 ** 2) / beta
            )
            self.indiff_func_x2 = lambda u0, x2, alpha, beta: np.sqrt(
                (u0 - beta * x2 ** 2) / alpha
            )
            self.cost_min_method = "numerical"
        elif self.preferences == "quasi_quasi":
            self.utility_func = lambda x1, x2, alpha, beta: x1 ** alpha * (x2 + beta)
            self.indiff_func_x1 = lambda u0, x1, alpha, beta: u0 / x1 ** alpha - beta
            self.indiff_func_x2 = lambda u0, x2, alpha, beta: (u0 / (x2 + beta)) ** (
                1 / alpha
            )
            self.utility_max_method = "numerical"
            self.cost_min_method = "numerical"
        elif self.preferences == "saturated":
            self.utility_func = lambda x1, x2, alpha, beta: -(
                (x1 - alpha) ** 2 + (x2 - beta) ** 2
            )
            self.indifference_curves_method = self.preferences
            self.utility_max_method = "numerical"
            self.cost_min_method = "numerical"
            self.monotone = False
        else:
            raise ValueError("unknown utility function")

        # b. figures
        if self.automatic_lims:
            self.x1max = self.x1_scale * self.I / self.p1
            self.x2max = self.x2_scale * self.I / self.p2

    def figure(self):

        fig = plt.figure(frameon=False, figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim([0, self.x1max])
        ax.set_ylim([0, self.x2max])
        ax.set_xlabel(self.x1_label)
        ax.set_ylabel(self.x2_label)

        fig.tight_layout()

        return fig, ax

    def legend(self, ax, **kwargs):

        kwargs.setdefault("loc", "upper right")
        kwargs.setdefault("frameon", True)

        legend = ax.legend(**kwargs)
        frame = legend.get_frame()
        frame.set_facecolor("white")

    ##########
    # choice #
    ##########

    def utility(self, x1, x2):

        return self.utility_func(x1, x2, self.alpha, self.beta)

    def maximize_utility(self, numerical=False, monotone=True, **kwargs):

        # a. update
        for key, val in kwargs.items():
            setattr(self, key, val)

        # b. solve
        if (self.utility_max_method == "numerical" and self.monotone == True) or (
            numerical and monotone
        ):

            # a. target
            def x2_func(x1):
                return (self.I - self.p1 * x1) / self.p2

            def target(x1):
                x2 = x2_func(x1)
                return -self.utility(x1, x2)

            # b. solve
            xopt = optimize.fminbound(target, 0, self.I / self.p1)

            # c. save
            self.x1_ast = xopt
            self.x2_ast = x2_func(self.x1_ast)

        elif self.utility_max_method == "numerical" or numerical:

            # a. target
            def target_2d(x):
                excess_spending = self.p1 * x[0] + self.p2 * x[1] - self.I
                return -self.utility(x[0], x[1]) + 1000 * np.max(
                    [excess_spending, -x[0], -x[1], 0]
                )

            # b. solve
            x0 = np.array([self.I / self.p1, self.I / self.p2]) / 2
            res = optimize.minimize(target_2d, x0, method="Nelder-Mead")

            # c. save
            self.x1_ast = res.x[0]
            self.x2_ast = res.x[1]

        elif self.preferences == "cobb_douglas":

            self.x1_ast = self.alpha / (self.alpha + self.beta) * self.I / self.p1
            self.x2_ast = self.beta / (self.alpha + self.beta) * self.I / self.p2

        elif self.preferences == "perfect_substitutes" or self.preferences == "concave":

            self.x1_ast = self.I / self.p1
            self.x2_ast = self.I / self.p2
            if self.x1_ast ** self.alpha < self.x2_ast ** self.beta:
                self.x1_ast = 0
            else:
                self.x2_ast = 0

        elif self.preferences == "perfect_complements":

            self.x1_ast = self.I / (self.p1 + self.alpha / self.beta * self.p2)
            self.x2_ast = self.I / (self.beta / self.alpha * self.p1 + self.p2)

        elif self.preferences == "quasi_log":

            self.x1_ast = (self.alpha * self.p2) / (self.beta * self.p1)
            self.x2_ast = (self.I - self.p1 * self.x1_ast) / self.p2
            if self.x2_ast < 0:
                self.x1_ast = self.I / self.p1
                self.x2_ast = 0

        elif self.preferences == "quasi_sqrt":

            self.x1_ast = (((self.alpha * self.p2) / (self.beta * self.p1)) / 2) ** 2
            self.x2_ast = (self.I - self.p1 * self.x1_ast) / self.p2
            if self.x2_ast < 0:
                self.x1_ast = self.I / self.p1
                self.x2_ast = 0

        else:
            raise ValueError("unknown solution method")

        # c. utility
        self.u_ast = self.utility(self.x1_ast, self.x2_ast)

        # d. return
        return np.array([self.x1_ast, self.x2_ast, self.u_ast])

    def plot_max(self, ax, **kwargs):

        kwargs.setdefault("ls", "")
        kwargs.setdefault("marker", "*")
        kwargs.setdefault("markersize", 7)
        kwargs.setdefault("color", "black")
        kwargs.setdefault(
            "label", f"$u({self.x1_ast:.2f},{self.x2_ast:.2f}) = {self.u_ast:.2f}$"
        )

        ax.plot(self.x1_ast, self.x2_ast, **kwargs)

    def minimize_cost(self, u, numerical=False, **kwargs):

        # a. update
        for key, val in kwargs.items():
            setattr(self, key, val)

        # b. solve
        if self.cost_min_method == "numerical" or numerical:

            def target_2d(x):
                x1 = x[0]
                x2 = x[1]
                udiff = (self.utility(x1, x2) - u) ** 2

                return (
                    self.p1 * x1
                    + self.p2 * x2
                    + 1000 * udiff
                    + 1000 * np.max([-x[0], -x[1], 0])
                )

            # b. solve
            x0 = np.array([self.I / self.p1, self.I / self.p2]) / 2
            res = optimize.minimize(target_2d, x0, method="Nelder-Mead")

            # c. save
            self.h1_ast = res.x[0]
            self.h2_ast = res.x[1]

        elif self.preferences == "cobb_douglas":

            self.h1_ast = (u * (self.p1 / self.p2) ** (-self.beta)) ** (
                1 / (self.alpha + self.beta)
            )
            self.h2_ast = (u * (self.p2 / self.p1) ** (-self.alpha)) ** (
                1 / (self.alpha + self.beta)
            )

        elif self.preferences == "quasi_log":

            self.h1_ast = (self.alpha * self.p2) / (self.beta * self.p1)
            self.h2_ast = (u - self.alpha * np.log(self.h1_ast)) / self.beta

        else:
            raise ValueError("unknown solution method")

        # c. cost
        self.h_cost = self.p1 * self.h1_ast + self.p2 * self.h2_ast

        # d. return
        return np.array([self.h1_ast, self.h2_ast, self.h_cost])

    #############
    # budgetset #
    #############

    def plot_budgetline(self, ax, **kwargs):

        kwargs.setdefault("color", "black")
        kwargs.setdefault("lw", 2)

        if self.budgetsettype == "kinked":

            x = [
                0,
                self.kink_point,
                self.kink_point
                + (self.I - self.p1 * self.kink_point) / (self.p1 - self.kink_slope),
            ]
            y = [self.I / self.p2, (self.I - self.p1 * self.kink_point) / self.p2, 0]

            ax.plot(x, y, **kwargs)

        else:

            x = [0, self.I / self.p1]
            y = [self.I / self.p2, 0]

            ax.plot(x, y, **kwargs)

    def plot_budgetset(self, ax, **kwargs):

        kwargs.setdefault("lw", 2)
        kwargs.setdefault("alpha", 0.5)

        if self.budgetsettype == "kinked":

            x = [
                0,
                0,
                self.kink_point,
                self.kink_point
                + (self.I - self.p1 * self.kink_point) / (self.p1 - self.kink_slope),
            ]
            y = [0, self.I / self.p2, (self.I - self.p1 * self.kink_point) / self.p2, 0]

            ax.fill(x, y, **kwargs)

        else:

            x = [0, 0, self.I / self.p1]
            y = [0, self.I / self.p2, 0]

            ax.fill(x, y, **kwargs)

    def plot_budgetline_slope(self, ax, scale_x=1.03, scale_y=1.03, **kwargs):

        x = (self.I / self.p1) / 2 * scale_x
        y = (self.I / self.p2) / 2 * scale_y

        ax.text(x, y, f"slope = -{self.p1/self.p2:.2f}", **kwargs)

    def plot_endowment(
        self, ax, scale_x=1.03, scale_y=1.03, text="endowment", **kwargs
    ):

        kwargs.setdefault("color", "black")

        ax.scatter(self.e1, self.e2, **kwargs)
        ax.text(self.e1 * scale_x, self.e2 * scale_y, text)

    #######################
    # indifference curves #
    #######################

    def find_indifference_curve_analytical(self, u0=None):

        if u0 == None:
            u0 = self.u_ast

        # a. fix x1
        x1_x1 = np.linspace(0, self.x1max, self.N)
        with np.errstate(divide="ignore", invalid="ignore"):
            x2_x1 = self.indiff_func_x1(u0, x1_x1, self.alpha, self.beta)

        # b. fix x2
        x2_x2 = np.linspace(0, self.x2max, self.N)
        with np.errstate(divide="ignore", invalid="ignore"):
            x1_x2 = self.indiff_func_x2(u0, x2_x2, self.alpha, self.beta)

        # c. combine
        x1 = np.hstack([x1_x1, x1_x2])
        x2 = np.hstack([x2_x1, x2_x2])

        # d. clean
        with np.errstate(divide="ignore", invalid="ignore"):
            u0s = self.utility(x1, x2)
        I = np.isclose(u0s, u0)
        x1 = x1[I]
        x2 = x2[I]

        # e. sort
        I = np.argsort(x1)
        x1 = x1[I]
        x2 = x2[I]

        return x1, x2

    def find_indifference_curve_numerical(self, u0=None):

        if u0 == None:
            u0 = self.u_ast

        x1 = []
        x2 = []

        # a. fix x1
        x1_x1 = np.linspace(0, self.x1max, self.N)
        for x1_now in x1_x1:

            def target_for_x2(x2):
                with np.errstate(divide="ignore", invalid="ignore"):
                    udiff = self.utility(x1_now, x2) - u0
                return udiff

            x_A, _infodict_A, ier_A, _mesg_A = optimize.fsolve(
                target_for_x2, 0, full_output=True
            )
            x_B, _infodict_B, ier_B, _mesg_B = optimize.fsolve(
                target_for_x2, self.x2max, full_output=True
            )

            if ier_A == 1:
                x1.append(x1_now)
                x2.append(x_A[0])
            else:
                x1.append(np.nan)
                x2.append(np.nan)

            if ier_B == 1 and np.abs(x_A[0] - x_B[0]) > 0.01:
                x1.append(x1_now)
                x2.append(x_B[0])
            else:
                x1.append(np.nan)
                x2.append(np.nan)

        # b. fix x2
        x2_x2 = np.linspace(0, self.x2max, self.N)
        for x2_now in x2_x2:

            def target_for_x1(x1):
                with np.errstate(divide="ignore", invalid="ignore"):
                    udiff = self.utility(x1, x2_now) - u0
                return udiff

            x_A, _infodict_A, ier_A, _mesg_A = optimize.fsolve(
                target_for_x1, 0, full_output=True
            )
            x_B, _infodict_B, ier_B, _mesg_B = optimize.fsolve(
                target_for_x1, self.x1max, full_output=True
            )

            if ier_A == 1:
                x1.append(x_A[0])
                x2.append(x2_now)
            else:
                x1.append(np.nan)
                x2.append(np.nan)

            if ier_B == 1 and np.abs(x_A[0] - x_B[0]) > 0.01:
                x1.append(x_B[0])
                x2.append(x2_now)
            else:
                x1.append(np.nan)
                x2.append(np.nan)

        # c. clean
        x1 = np.array(x1)
        x2 = np.array(x2)
        with np.errstate(divide="ignore", invalid="ignore"):
            u0s = self.utility(x1, x2)
        I = np.isclose(u0s, u0)
        x1 = x1[I]
        x2 = x2[I]

        # d. sort
        I = np.argsort(x1)
        x1 = x1[I]
        x2 = x2[I]

        return x1, x2

    def find_indifference_curve(self, u0=None, numerical=False):

        if (
            self.indifference_curves_method == "numerical"
            or self.indifference_curves_method == "saturated"
            or numerical
        ):
            x1, x2 = self.find_indifference_curve_numerical(u0)
        elif self.indifference_curves_method == "perfect_complements":
            corner_x1 = u0 / self.alpha
            x1 = [corner_x1, corner_x1, self.x1max]
            corner_x2 = u0 / self.beta
            x2 = [self.x2max, corner_x2, corner_x2]
        else:
            x1, x2 = self.find_indifference_curve_analytical(u0)

        return x1, x2

    def plot_indifference_curves(
        self, ax, u0s=[], do_label=True, only_dots=False, numerical=False, **kwargs
    ):

        kwargs.setdefault("lw", 2)

        # a. find utility levels
        if len(u0s) == 0:
            self.maximize_utility()
            u0s = [
                self.u_ast,
                self.utility(0.8 * self.x1_ast, 0.8 * self.x2_ast),
                self.utility(1.2 * self.x1_ast, 1.2 * self.x2_ast),
            ]

        # b. construct and plot indifference curves
        for i, u0 in enumerate(u0s):

            if self.preferences == "saturated":
                radius = np.sqrt(-u0)
                circle = plt.Circle(
                    (self.alpha, self.beta),
                    radius,
                    fill=False,
                    color=colors[i],
                    **kwargs,
                )
                ax.add_artist(circle)
                continue
            else:
                x1, x2 = self.find_indifference_curve(u0, numerical)

            if only_dots:
                ax.scatter(x1, x2, 1, **kwargs)
            else:
                if do_label:
                    ax.plot(x1, x2, label=f"$u = {u0:.2f}$", **kwargs)
                else:
                    ax.plot(x1, x2, **kwargs)

    def plot_monotonicity_check(self, ax, x1=None, x2=None, text="north-east"):

        # a. value
        if x1 == None:
            x1 = self.x1_ast
        if x2 == None:
            x2 = self.x2_ast

        # b. plot
        polygon = plt.Polygon(
            [
                [x1, x2],
                [self.x1max, x2],
                [self.x1max, self.x2max],
                [x1, self.x2max],
                [x1, x2],
            ],
            color="black",
            alpha=0.10,
        )
        ax.add_patch(polygon)
        ax.text(x1 * 1.05, 0.95 * self.x2max, text)

    def plot_convexity_check(self, ax, u=None, text="mix"):

        # a. value
        if u == None:
            u = self.u_ast

        # b. indifference curve
        x1, x2 = self.find_indifference_curve_analytical(u)

        # c. select
        I = (np.isnan(x1) == False) & (np.isnan(x2) == False)
        J = (x1[I] > 0) & (x2[I] > 0) & (x1[I] < self.x1max) & (x2[I] < self.x2max)

        x1 = x1[I][J]
        x2 = x2[I][J]
        N = x1.size
        i_low = np.int(N / 3)
        i_high = np.int(N * 2 / 3)

        x1_low = x1[i_low]
        x2_low = x2[i_low]

        x1_high = x1[i_high]
        x2_high = x2[i_high]

        # d. plot
        ax.plot(
            [x1_low, x1_high],
            [x2_low, x2_high],
            ls="--",
            marker="o",
            markerfacecolor="none",
            color="black",
        )

        x1 = 0.05 * x1_low + 0.95 * x1_high
        x2 = 0.05 * x2_low + 0.95 * x2_high
        ax.text(x1, x2 * 1.05, text)

    ##############################
    # price change decomposition #
    ##############################

    def plot_decomposition_exogenous(self, ax, p1_old, p1_new, p2=1, steps=3):

        p1_saved, p2_saved = self.p1, self.p2

        # a. calculations
        A = self.maximize_utility(p1=p1_old, p2=p2)
        x1, x2, u_ast = A

        C = self.maximize_utility(p1=p1_new, p2=p2)
        x1_new, x2_new, u_ast_new = C

        B = self.minimize_cost(u_ast, p1=p1_new, p2=p2)
        h1, h2, h_cost = B

        # b. plots
        self.p1, self.p2 = p1_old, p2
        self.plot_budgetline(ax, ls="-", label="original", color=colors[0])
        if steps > 1:
            self.p1, self.p2 = p1_new, p2
            self.plot_budgetline(ax, ls="-", label="final", color=colors[1])
        if steps > 2:
            self.p1, self.p2, self.I = p1_new, p2, h_cost
            self.plot_budgetline(
                ax, ls="-", alpha=0.50, label="compensated", color=colors[2]
            )

        # A
        ax.plot(x1, x2, "ro", color="black")
        ax.text(x1 * 1.03, x2 * 1.03, "$A$")
        self.plot_indifference_curves(
            ax, [u_ast], do_label=False, ls="--", color=colors[0]
        )

        # B
        if steps > 2:
            ax.plot(h1, h2, "ro", color="black")
            ax.text(h1 * 1.03, h2 * 1.03, "$B$")

        # C
        if steps > 1:
            ax.plot(x1_new, x2_new, "ro", color="black")
            ax.text(x1_new * 1.03, x2_new * 1.03, "$C$")
            self.plot_indifference_curves(
                ax, [u_ast_new], do_label=False, ls="--", color=colors[1]
            )

        if steps > 2:
            line = f"subtitution: $B-A$ = ({h1-x1:5.2f},{h2-x2:5.2f})\n"
            line += f"income: $C-B$ = ({x1_new-h1:5.2f},{x2_new-h2:5.2f})\n"
            ax.text(0.55 * self.x1max, 0.87 * self.x2max, line, backgroundcolor="white")

        self.p1, self.p2 = p1_saved, p2_saved
        return A, B, C

    def plot_decomposition_endogenous(self, ax, p1_old, p1_new, p2, e1, e2, steps=4):

        p1_saved, p2_saved, e1_saved, e2_saved = self.p1, self.p2, self.e1, self.e2

        I_old = p1_old * e1 + p2 * e2
        I_new = p1_new * e1 + p2 * e2

        # a. calculations
        A = self.maximize_utility(p1=p1_old, p2=p2, I=I_old)
        x1, x2, u_max = A

        C1 = self.maximize_utility(p1=p1_new, p2=p2, I=I_old)
        x1_fixI, x2_fixI, u_max_fixI = C1

        B = self.minimize_cost(u_max, p1=p1_new, p2=p2)
        h1, h2, h_cost = B

        C2 = self.maximize_utility(p1=p1_new, p2=p2, I=I_new)
        x1_new, x2_new, u_max_new = C2

        # b. plots
        self.p1, self.p2, self.I = p1_old, p2, I_old
        self.plot_budgetline(ax, ls="-", label="original", color=colors[0])
        if steps > 1:
            self.p1, self.p2, self.I = p1_new, p2, I_new
            self.plot_budgetline(ax, ls="-", label="final", color=colors[1])
        if steps > 2:
            self.p1, self.p2, self.I = p1_new, p2, h_cost
            self.plot_budgetline(
                ax, ls="-", alpha=0.50, label="compensated", color=colors[2]
            )
        if steps > 3:
            self.p1, self.p2, self.I = p1_new, p2, I_old
            self.plot_budgetline(ax, ls="-", label="constant income", color=colors[3])
        ax.plot(e1, e2, "ro", color="black")
        ax.text(e1 * 1.03, e2 * 1.03, "$E$")

        # A
        ax.plot(x1, x2, "ro", color="black")
        ax.text(x1 * 1.03, x2 * 1.03, "$A$")
        self.plot_indifference_curves(
            ax, [u_max], do_label=False, ls="--", color=colors[0]
        )

        # B
        if steps > 2:
            ax.plot(h1, h2, "ro", color="black")
            ax.text(h1 * 1.03, h2 * 1.03, "$B$")

        # C2
        if steps > 1:
            ax.plot(x1_new, x2_new, "ro", color="black")
            ax.text(x1_new * 1.03, x2_new * 1.03, "$C_2$")
            self.plot_indifference_curves(
                ax, [u_max_new], do_label=False, ls="--", color=colors[1]
            )

        # C1
        if steps > 3:
            ax.plot(x1_fixI, x2_fixI, "ro", color="black")
            ax.text(x1_fixI * 1.03, x2_fixI * 1.03, f"$C_1$")
            self.plot_indifference_curves(
                ax, [u_max_fixI], do_label=False, ls="--", color=colors[3]
            )

        if steps > 3:
            line = f"subtitution: $B-A$ = ({h1-x1:5.2f},{h2-x2:5.2f})\n"
            line += f"income: $C_1-B$ = ({x1_fixI-h1:5.2f},{x2_fixI-h2:5.2f})\n"
            line += f"wealth: $C_2-C_1$ = ({x1_new-x1_fixI:5.2f},{x2_new-x2_fixI:5.2f})"
            ax.text(0.55 * self.x1max, 0.87 * self.x2max, line, backgroundcolor="white")

        self.p1, self.p2, self.e1, self.e2 = p1_saved, p2_saved, e1_saved, e2_saved
        return A, B, C1, C2


#######################
# interactive figures #
#######################


def _interactive_budgetset_exogenous(p1, p2, I, par):

    consumer = ConsumerClass(
        p1=p1,
        p2=p2,
        I=I,
        budgetsettype="exogenous",
        x1max=par["x1max"],
        x2max=par["x2max"],
    )
    _fig, ax = consumer.figure()

    consumer.plot_budgetline(ax, color="black")
    consumer.plot_budgetset(ax)
    consumer.plot_budgetline_slope(ax)


def _interactive_budgetset_endogenous(p1, p2, e1, e2, par):

    consumer = ConsumerClass(
        p1=p1,
        p2=p2,
        e1=e1,
        e2=e2,
        budgetsettype="endogenous",
        x1max=par["x1max"],
        x2max=par["x2max"],
    )
    _fig, ax = consumer.figure()

    consumer.plot_budgetline(ax, color="black")
    consumer.plot_budgetset(ax)
    consumer.plot_budgetline_slope(ax)
    consumer.plot_endowment(ax)


def _interactive_budgetset_kink(p1, kink_point, kink_slope, p2, I, par):

    show_warning = False
    if p1 - kink_slope <= 0:
        kink_slope = p1 - 1e-8
        show_warning = True

    consumer = ConsumerClass(
        p1=p1,
        p2=p2,
        I=I,
        budgetsettype="kinked",
        kink_point=kink_point,
        kink_slope=kink_slope,
        x1max=par["x1max"],
        x2max=par["x2max"],
    )
    _fig, ax = consumer.figure()

    if show_warning:
        ax.text(
            0.65 * consumer.x1max,
            0.95 * consumer.x2max,
            f"warning: $p_1-\\Delta_1 <= 0$",
            backgroundcolor="white",
        )

    consumer.plot_budgetline(ax)
    consumer.plot_budgetset(ax)


def interactive_budgetset(budgetsettype, **kwargs):

    kwargs.setdefault("x1max", 10)
    kwargs.setdefault("x2max", 10)

    if budgetsettype == "exogenous":

        widgets.interact(
            _interactive_budgetset_exogenous,
            p1=widgets.FloatSlider(
                description="$p_1$", min=0.1, max=5, step=0.05, value=2
            ),
            p2=widgets.FloatSlider(
                description="$p_2$", min=0.1, max=5, step=0.05, value=1
            ),
            I=widgets.FloatSlider(
                description="$I$", min=0.1, max=20, step=0.10, value=5
            ),
            par=widgets.fixed(kwargs),
        )

    elif budgetsettype == "endogenous":

        widgets.interact(
            _interactive_budgetset_endogenous,
            p1=widgets.FloatSlider(
                description="$p_1$", min=0.1, max=5, step=0.05, value=2
            ),
            p2=widgets.FloatSlider(
                description="$p_2$", min=0.1, max=5, step=0.05, value=1
            ),
            e1=widgets.FloatSlider(
                description="$e_1$", min=0.0, max=5, step=0.05, value=3
            ),
            e2=widgets.FloatSlider(
                description="$e_2$", min=0.0, max=5, step=0.05, value=4
            ),
            par=widgets.fixed(kwargs),
        )

    elif budgetsettype == "kinked":

        widgets.interact(
            _interactive_budgetset_kink,
            p1=widgets.FloatSlider(
                description="$p_1$", min=0.1, max=5, step=0.05, value=1
            ),
            kink_point=widgets.FloatSlider(
                description="$\\overline{x}_1$", min=0.1, max=5, step=0.05, value=2
            ),
            kink_slope=widgets.FloatSlider(
                description="$\\Delta_1$", min=-5, max=5, step=0.10, value=-1
            ),
            p2=widgets.FloatSlider(
                description="$p_2$", min=0.1, max=5, step=0.05, value=1
            ),
            I=widgets.FloatSlider(
                description="$I$", min=0.10, max=20, step=0.10, value=10
            ),
            par=widgets.fixed(kwargs),
        )


def interactive_utility_settings(preferences, kwargs):

    # a. special
    if preferences == "ces":

        kwargs.setdefault("alpha", 0.65)
        kwargs.setdefault("beta", 0.85)

        kwargs.setdefault("alpha_min", 0.05)
        kwargs.setdefault("alpha_max", 0.99)

        kwargs.setdefault("beta_min", -0.95)
        kwargs.setdefault("beta_max", 10.01)

    elif preferences == "saturated":

        kwargs.setdefault("alpha", 5.00)
        kwargs.setdefault("beta", 5.00)

        kwargs.setdefault("alpha_min", 0.0)
        kwargs.setdefault("alpha_max", 8)

        kwargs.setdefault("beta_min", 0.0)
        kwargs.setdefault("beta_max", 8)

    # b. standard
    kwargs["preferences"] = preferences

    kwargs.setdefault("x1max", 10)
    kwargs.setdefault("x2max", 10)
    kwargs.setdefault("show_monotonicity_check", False)
    kwargs.setdefault("show_convexity_check", False)

    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("alpha_min", 0.05)
    kwargs.setdefault("alpha_max", 4.00)
    kwargs.setdefault("alpha_step", 0.05)

    kwargs.setdefault("beta", 1)
    kwargs.setdefault("beta_min", 0.05)
    kwargs.setdefault("beta_max", 4.00)
    kwargs.setdefault("beta_step", 0.05)

    kwargs.setdefault("x1s", [2, 3, 4])
    kwargs.setdefault("x2s", [2, 3, 4])

    kwargs.setdefault("N", 100)

    kwargs.setdefault("p1", 1)
    kwargs.setdefault("p1_new", 3)
    kwargs.setdefault("p2", 1)
    kwargs.setdefault("I", 8)

    kwargs.setdefault("e1", 4)
    kwargs.setdefault("e2", 2)

    kwargs.setdefault("p1_min", 0.05)
    kwargs.setdefault("p1_max", 4.00)
    kwargs.setdefault("p1_step", 0.05)

    kwargs.setdefault("p2_min", 0.05)
    kwargs.setdefault("p2_max", 4.00)
    kwargs.setdefault("p2_step", 0.05)

    kwargs.setdefault("I_min", 0.5)
    kwargs.setdefault("I_max", 20)
    kwargs.setdefault("I_step", 0.05)

    kwargs.setdefault("e1_min", 0.0)
    kwargs.setdefault("e1_max", 5)
    kwargs.setdefault("e1_step", 0.05)

    kwargs.setdefault("e2_min", 0.0)
    kwargs.setdefault("e2_max", 5)
    kwargs.setdefault("e2_step", 0.05)


def _interactive_indifference_curves(alpha, beta, par):

    par["alpha"] = alpha
    par["beta"] = beta
    consumer = ConsumerClass(**par)
    _fig, ax = consumer.figure()

    # 45 degrees
    ax.plot(
        [0, consumer.x1max],
        [0, consumer.x2max],
        "--",
        color="black",
        zorder=1,
        alpha=0.1,
    )

    # indifference curves
    x1s = par["x1s"]
    x2s = par["x2s"]
    us = [consumer.utility(x1, x2) for x1, x2 in zip(x1s, x2s)]
    [ax.plot(x1, x2, "ro", color="black") for x1, x2 in zip(x1s, x2s)]
    [ax.text(x1 * 1.03, x2 * 1.03, f"u = {u:5.2f}") for x1, x2, u in zip(x1s, x2s, us)]
    consumer.plot_indifference_curves(ax, us)

    if par["show_monotonicity_check"]:
        consumer.plot_monotonicity_check(ax, x1=x1s[1], x2=x2s[1])
    if par["show_convexity_check"]:
        consumer.plot_convexity_check(ax, u=us[1])


def interactive_indifference_curves(preferences="cobb_douglas", **kwargs):

    interactive_utility_settings(preferences, kwargs)

    widgets.interact(
        _interactive_indifference_curves,
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
        par=widgets.fixed(kwargs),
    )


def _interactive_utility_max(p1, p2, I, alpha, beta, gamma, par):

    par["p1"] = p1
    par["p2"] = p2
    par["I"] = I
    par["alpha"] = alpha
    par["beta"] = beta
    consumer = ConsumerClass(**par)
    _fig, ax = consumer.figure()

    def xs_from_gamma(gamma):
        x1 = I / p1 * gamma
        x2 = (I - p1 * x1) / p2
        return x1, x2

    x1, x2 = xs_from_gamma(gamma)

    # a. calculations
    x1_max, x2_max, u_max = consumer.maximize_utility()

    u_alt = [consumer.utility(x1, x2), consumer.utility(x1_max * 1.2, x2_max * 1.2)]

    # c. plots
    consumer.plot_max(ax)
    ax.text(x1_max * 1.03, x2_max * 1.03, f"$u^{{max}} = {u_max:.2f}$")

    ax.plot(x1, x2, "o", color=colors[1])
    ax.text(x1 * 1.03, x2 * 1.03, f'$u^{{\\gamma}} = {consumer.utility(x1,x2):.2f}$')

    consumer.plot_indifference_curves(ax, [u_max])
    consumer.plot_indifference_curves(ax, u_alt, ls="--")

    consumer.plot_budgetline(ax)


def interactive_utility_max(preferences="cobb_douglas", **kwargs):

    interactive_utility_settings(preferences, kwargs)

    widgets.interact(
        _interactive_utility_max,
        p1=widgets.FloatSlider(
            description="$p_1$",
            min=kwargs["p1_min"],
            max=kwargs["p1_max"],
            step=kwargs["p1_step"],
            value=kwargs["p1"],
        ),
        p2=widgets.FloatSlider(
            description="$p_2$",
            min=kwargs["p2_min"],
            max=kwargs["p2_max"],
            step=kwargs["p2_step"],
            value=kwargs["p2"],
        ),
        I=widgets.FloatSlider(
            description="$I$",
            min=kwargs["I_min"],
            max=kwargs["I_max"],
            step=kwargs["I_step"],
            value=kwargs["I"],
        ),
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
            description="$\\gamma$", min=0.01, max=0.99, step=0.01, value=0.25
        ),
        par=widgets.fixed(kwargs),
    )


def _interactive_slutsky_exogenous(par, steps, p1_old, p1_new, alpha, beta):

    par["alpha"] = alpha
    par["beta"] = beta

    consumer = ConsumerClass(**par)
    _fig, ax = consumer.figure()

    p2 = 1
    consumer.plot_decomposition_exogenous(ax, p1_old, p1_new, p2, steps=steps)


def interactive_slutsky_exogenous(preferences="cobb_douglas", **kwargs):

    interactive_utility_settings(preferences, kwargs)

    widgets.interact(
        _interactive_slutsky_exogenous,
        par=widgets.fixed(kwargs),
        steps=widgets.IntSlider(description="steps", min=1, max=3, step=1, value=1),
        p1_old=widgets.FloatSlider(
            description="$p_1$",
            min=kwargs["p1_min"],
            max=kwargs["p1_max"],
            step=kwargs["p1_step"],
            value=kwargs["p1"],
        ),
        p1_new=widgets.FloatSlider(
            description="$p_1^{\\prime}$",
            min=kwargs["p1_min"],
            max=kwargs["p1_max"],
            step=kwargs["p1_step"],
            value=kwargs["p1_new"],
        ),
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
    )


def _interactive_slutsky_endogenous(par, steps, p1_old, p1_new, e1, e2, alpha, beta):

    par["alpha"] = alpha
    par["beta"] = beta

    consumer = ConsumerClass(**par)
    _fig, ax = consumer.figure()

    p2 = 1
    consumer.plot_decomposition_endogenous(ax, p1_old, p1_new, p2, e1, e2, steps=steps)


def interactive_slutsky_endogenous(preferences="cobb_douglas", **kwargs):

    interactive_utility_settings(preferences, kwargs)

    widgets.interact(
        _interactive_slutsky_endogenous,
        par=widgets.fixed(kwargs),
        steps=widgets.IntSlider(description="steps", min=1, max=4, step=1, value=1),
        p1_old=widgets.FloatSlider(
            description="$p_1$",
            min=kwargs["p1_min"],
            max=kwargs["p1_max"],
            step=kwargs["p1_step"],
            value=kwargs["p1"],
        ),
        p1_new=widgets.FloatSlider(
            description="$p_1^{\\prime}$",
            min=kwargs["p1_min"],
            max=kwargs["p1_max"],
            step=kwargs["p1_step"],
            value=kwargs["p1_new"],
        ),
        e1=widgets.FloatSlider(
            description="$e_1$",
            min=kwargs["e1_min"],
            max=kwargs["e1_max"],
            step=kwargs["e1_step"],
            value=kwargs["e1"],
        ),
        e2=widgets.FloatSlider(
            description="$e_2$",
            min=kwargs["e2_min"],
            max=kwargs["e2_max"],
            step=kwargs["e2_step"],
            value=kwargs["e2"],
        ),
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
    )
