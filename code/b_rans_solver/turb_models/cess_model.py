#**************************************************************************
#       Implementation of the algebraic Cess model 
#       Reference,
#       Cess, R.D., "A survery of the literature on heat transfer in 
#       turbulent tube flow", Tech. Rep. 8-0529-R24, Westinghouse, 1958. 
#       - Modified for channel flow by: Hussain, A.K.M.F. and Reynolds
#       W.C., "Measurements in fully developed turbulent channel flow",
#       ASME Journal of fluid eng., 1975.
#**************************************************************************
# Cess developed a analytical expression based on the effective viscosity 
# mueff=(mu+ mut). The expression combines the near wall behaviour of the
# eddy viscosity developed by van Driest and the outer layer behaviour 
# from Reichardt.
#
# Conventional model:
#    mueff/mu = 1/2*(1+1/9*kappa^2*ReT^2*(t1*t2)*[1-exp(-yplus/A)]^2)^(1/2) 
#               - 1/2;
#
# Otero et.al model:
#    mueff/mu = 1/2*(1+1/9*kappa^2*Rets^2*(t1*t2)*[1-exp(-ystar/A)]^2)^(1/2) 
#               - 1/2;
# This models uses "yplus" and "ReT". It must be replace by its semi-locally
# scaled counter part "ystar" and "Rets", respectively
#
#
# Input:
#   r           density
#   mu          molecular viscosity
#   ReT         friction Reynolds number ReT=utau r_wall h/ mu_wall
#   mesh        mesh structure
#   compFlag    flag to solve the model with compressible modifications
#
# Output:
#   mut         eddy viscosity or turbulent viscosity
#

# ------------------------------------------------------------------------
#                  Basic Libraries
# ------------------------------------------------------------------------

from    os.path       import  abspath, dirname
from    sys           import  path              as  sys_path

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(dirname(abspath(__file__))))
from  cfd_solver  import  CFD_solver
sys_path.pop()

# ------------------------------------------------------------------------
#                  Utility functions
# ------------------------------------------------------------------------

power      =  lambda a  , b   : a**b

# ------------------------------------------------------------------------
#                  Cess turbulence model
# ------------------------------------------------------------------------

class CFD_Solver_Cess(CFD_solver):
    def __init__(self, args):
        super().__init__(args)
        assert hasattr(self, 'args')
        self.Ret             =  args['Ret']
        self.tag_rans_model  =  'Cess'

        # compressibility correction of (Otero et al. 2018)
        assert not args['compressible_correction']  # not enabled for this study
    
    def turb_model(self):
        r     =  self.rans_rho_molec
        mu    =  self.rans_mu_molec
        ReTau =  self.Ret

        torch      =  self.args['torch']
        sqrt, exp  =  torch.sqrt, torch.exp

        d          =  self.y

        # Model constants
        kappa    =  0.426
        A        =  25.4

        ReTauArr =  sqrt(r/r[0])/(mu/mu[0])*ReTau
        yplus    =  d*ReTauArr

        # ReTauArr = np.sqrt(r/r[0])/(mu/mu[0])*ReTau
        # yplus = d*ReTauArr
        # if compressibleCorrection == 1:
        #     ReTauArr = np.sqrt(r/r[0])/(mu/mu[0])*ReTau
        #     yplus = d*ReTauArr
        # else: 
        #     ReTauArr = np.ones(self.n)*ReTau
        #     yplus = d*ReTauArr

        df   = 1 - exp(-yplus/A)
        t1   =  power(2*d-d*d, 2)
        t2   =  power(3-4*d+2*d*d, 2)
        mut  =  mu* ( 0.5*power(1 + 1/9*power(kappa*ReTauArr, 2)*(t1*t2)*df*df, 0.5) - 0.5 )
        
        self.rans_mu_turb = mut

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    sys_path.append(dirname(dirname(abspath(__file__))))
    from  cfd_solver  import  quick_benchmark
    sys_path.pop()

    quick_benchmark(CFD_Solver_Cess)