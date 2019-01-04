# note: documentation not written yet

import numpy as np

import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
import ipywidgets as widgets

from .consumption import ConsumerClass

class EdgeworthModel():

    def __init__(self,consumer_A,consumer_B,eA=[1,1],eB=[1,1],**kwargs):

        # a. baseline setup

        # endowment        
        self.e1bar = eA[0]+eB[0]
        self.e2bar = eA[1]+eB[1]

        # prices
        self.p1 = 1
        self.p2 = 1

        # misc
        self.N = 100

        # b. update 
        for key,val in kwargs.items():
            setattr(self,key,val)

        # c. consumers
        self.consumer_A = ConsumerClass(**consumer_A,
            budgetsettype='endogenous',p1=self.p1,p2=self.p2,e1=eA[0],e2=eA[1],
            x1max=self.e1bar,x2max=self.e2bar)
        self.consumer_B = ConsumerClass(**consumer_B,
            budgetsettype='endogenous',p1=self.p1,p2=self.p2,e1=eB[0],e2=eB[1],
            x1max=self.e1bar,x2max=self.e2bar)

    ##########
    # figure #
    ##########

    def figure(self):

        fig = plt.figure(frameon=False,figsize=(6,6),dpi=100)
        ax_A = fig.add_subplot(1,1,1)
        
        ax_A.set_xlim([-0.1,self.e1bar+0.1])
        ax_A.set_ylim([-0.1,self.e2bar+0.1])

        ax_A.set_xlabel('$x_1^A$')
        ax_A.set_ylabel('$x_2^A$')

        temp = ax_A.twinx()
        temp.set_ylabel('$x_2^B$')
        ax_B = temp.twiny()
        ax_B.set_xlabel('$x_1^B$')
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        ax_B.set_xlim([self.e1bar+0.1,-0.1])
        ax_B.set_ylim([self.e2bar+0.1,-0.1])

        fig.tight_layout()

        return fig,ax_A,ax_B

    def legend(self,ax,**kwargs):

        kwargs.setdefault('loc','upper right')
        kwargs.setdefault('frameon',True)

        legend = ax.legend(**kwargs)
        frame = legend.get_frame()
        frame.set_facecolor('white')  

    def pareto_figure(self,xAs):

        fig,ax_A,ax_B = self.figure()

        for i,(x1A,x2A) in enumerate(xAs):

            # A
            uA = self.consumer_A.utility(x1A,x2A)
            indiff_x1_A,indiff_x2_A = self.consumer_A.find_indifference_curve(u0=uA)
            if i == 0:
                ax_A.plot(indiff_x1_A,indiff_x2_A,color=colors[0],label='consumer A')
            else:
                ax_A.plot(indiff_x1_A,indiff_x2_A,color=colors[0])

            # B
            x1B = self.e1bar-x1A
            x2B = self.e2bar-x2A
            uB = self.consumer_B.utility(x1B,x2B)
            indiff_x1_B,indiff_x2_B = self.consumer_B.find_indifference_curve(u0=uB)
            if i == 0:
                ax_A.plot([],[],color=colors[1],label='consumer B')
                ax_B.plot(indiff_x1_B,indiff_x2_B,color=colors[1],label='consumer B')
            else:
                ax_B.plot(indiff_x1_B,indiff_x2_B,color=colors[1])
                            

        # box
        ax_A.plot([0,self.e1bar,self.e1bar,0,0],[0,0,self.e2bar,self.e2bar,0],lw='2',color='black') 

        return fig,ax_A,ax_B

    def walras_figure(self,xAs):

        _fig,ax_A,ax_B = self.pareto_figure(xAs)  

        # a. consumer A
        self.consumer_A.maximize_utility()
        self.consumer_A.plot_endowment(ax_A,text='E')
        self.consumer_A.plot_budgetline(ax_A)
        self.consumer_A.plot_max(ax_A,color=colors[0])
        ax_A.text(self.consumer_A.x1_ast*1.03,self.consumer_A.x2_ast*1.03,f'A')
        self.consumer_A.plot_indifference_curves(ax_A,u0s=[self.consumer_A.u_ast],color=colors[0])
        self.consumer_A.plot_indifference_curves(ax_A,color=colors[0],ls='--',lw=1)

        # b. consumer B
        self.consumer_B.maximize_utility()
        self.consumer_B.plot_max(ax_B,color=colors[1])
        ax_B.text(self.consumer_B.x1_ast*0.97,self.consumer_B.x2_ast*0.97,f'B')
        self.consumer_B.plot_indifference_curves(ax_B,u0s=[self.consumer_B.u_ast],color=colors[1])
        self.consumer_B.plot_indifference_curves(ax_B,color=colors[1],ls='--',lw=1)

def interactive_settings(preferences_A,preferences_B,kwargs):

    # a. special A
    if preferences_A == 'ces':

        kwargs.setdefault('alpha_A',0.50)
        kwargs.setdefault('alpha_A_min',0.05)
        kwargs.setdefault('alpha_A_max',0.99)

        kwargs.setdefault('beta_A',0.85)
        kwargs.setdefault('beta_A_min',-0.95)
        kwargs.setdefault('beta_A_max',10.01)

    # b. special B
    if preferences_B == 'ces':

        kwargs.setdefault('alpha_B',0.50)
        kwargs.setdefault('alpha_B_min',0.05)
        kwargs.setdefault('alpha_B_max',0.99)

        kwargs.setdefault('beta_B',0.85)
        kwargs.setdefault('beta_B_min',-0.95)
        kwargs.setdefault('beta_B_max',10.01)

    # c. standard    
    kwargs.setdefault('preferences_A',preferences_A)
    kwargs.setdefault('preferences_B',preferences_B)

    kwargs.setdefault('x1max',1)
    kwargs.setdefault('x2max',1)

    kwargs.setdefault('alpha_A',1.00)
    kwargs.setdefault('alpha_A_min',0.05)
    kwargs.setdefault('alpha_A_max',4.00)
    kwargs.setdefault('alpha_A_step',0.05)

    kwargs.setdefault('beta_A',1.50)
    kwargs.setdefault('beta_A_min',0.05)
    kwargs.setdefault('beta_A_max',4.00)
    kwargs.setdefault('beta_A_step',0.05)

    kwargs.setdefault('alpha_B',1.00)
    kwargs.setdefault('alpha_B_min',0.05)
    kwargs.setdefault('alpha_B_max',4.00)
    kwargs.setdefault('alpha_B_step',0.05)

    kwargs.setdefault('beta_B',1.50)
    kwargs.setdefault('beta_B_min',0.05)
    kwargs.setdefault('beta_B_max',4.00)
    kwargs.setdefault('beta_B_step',0.05)

    kwargs.setdefault('p1',2)
    kwargs.setdefault('p1_min',0.05   )
    kwargs.setdefault('p1_max',4.00    )
    kwargs.setdefault('p1_step',0.05)
    kwargs.setdefault('p2',1)

    kwargs.setdefault('omega_1',0.80)
    kwargs.setdefault('omega_2',0.20)

def _interactive_walras(p1,omega_1,omega_2,alpha_A,beta_A,alpha_B,beta_B,par):

    consumer_A = {}
    consumer_A['preferences'] = par['preferences_A']
    consumer_A['alpha'] = alpha_A
    consumer_A['beta'] = beta_A

    consumer_B = {}
    consumer_B['preferences'] = par['preferences_B']
    consumer_B['alpha'] = alpha_B
    consumer_B['beta'] = beta_B

    model = EdgeworthModel(consumer_A,consumer_B,p1=p1,p2=par['p2'],eA=(omega_1,omega_2),eB=(1-omega_1,1-omega_2))
    model.walras_figure(xAs=[])

def interactive_walras(preferences_A='cobb_douglas',preferences_B='cobb_douglas',**kwargs):

    interactive_settings(preferences_A,preferences_B,kwargs)

    widgets.interact(_interactive_walras,
        p1=widgets.BoundedFloatText(description='$p_1$',min=kwargs['p1_min'],max=kwargs['p1_max'],
            step=kwargs['p1_step'],value=kwargs['p1']),
        omega_1=widgets.FloatSlider(description='$\\omega_1$',min=0.05,max=0.99,
            step=0.05,value=kwargs['omega_1']),
        omega_2=widgets.FloatSlider(description='$\\omega_2$',min=0.05,max=0.99,
            step=0.05,value=kwargs['omega_2']),
        alpha_A=widgets.FloatSlider(description='$\\alpha^A$',min=kwargs['alpha_A_min'],max=kwargs['alpha_A_max'],
            step=kwargs['alpha_A_step'],value=kwargs['alpha_A']),
        beta_A=widgets.FloatSlider(description='$\\beta^A$',min=kwargs['beta_A_min'],max=kwargs['beta_A_max'],
            step=kwargs['beta_A_step'],value=kwargs['beta_A']),
        alpha_B=widgets.FloatSlider(description='$\\alpha^B$',min=kwargs['alpha_B_min'],max=kwargs['alpha_B_max'],
            step=kwargs['alpha_B_step'],value=kwargs['alpha_B']),
        beta_B=widgets.FloatSlider(description='$\\beta^B$',min=kwargs['beta_B_min'],max=kwargs['beta_B_max'],
            step=kwargs['beta_B_step'],value=kwargs['beta_B']),
        par=widgets.fixed(kwargs)
    )