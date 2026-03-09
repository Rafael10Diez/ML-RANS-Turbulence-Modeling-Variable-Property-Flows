#**************************************************************************
#       Implementation of k-epsilon MK model
#       Reference,
#       Myong, H.K. and Kasagi, N., "A new approach to the improvement of
#       k-epsilon turbulence models for wall bounded shear flow", JSME 
#       International Journal, 1990. 
#**************************************************************************
# An improved near-wall k-epsilon turbulence model that considers two 
# characteristics lenght scale for dissipation rate.
#
# Conventional models without compressible modifications:
#    k-eq:  0 = Pk - rho e + ddy[(mu+mut/sigma_k) dkdy]
#    e-eq:  0 = C_e1 f1 e/k Pk - rho C_e2 f2 e^2/k + ddy[(mu+mut/sigma_e)dedy] 
#
# Otero et.al model:
#    k-eq:  0 = Pk - rho e
#               + 1/sqrt(rho) ddy[1/sqrt(rho) (mu+mut/sigma_k) d(rho k)dy]
#    e-eq:  0 = C_e1 f1 e/k Pk - rho C_e2 f2 e^2/k 
#               + 1/rho ddy[1/sqrt(rho) (mu+mut/sigma_e) d(rho^1.5 e)dy] 
# This models uses "yplus". It must be replace by its semi-locally scaled
# counter part "ystar"
#
# Catris, S. and Aupoix, B., "Density corrections for turbulence
#       models", Aerosp. Sci. Techn., 2000.
#    k-eq:  0 = Pk - rho e 
#               + ddy[1/rho (mu+mut/sigma_k) d(rho k)dy]
#    e-eq:  0 = C_e1 f1 e/k Pk - rho C_e2 f2 e^2/k 
#               + 1/rho ddy[1/sqrt(rho) (mu+mut/sigma_e) d(rho^1.5 e)dy]
#
# For simplicty, the extra density factors of the Otero et.al and Catris/Aupoix  
# models were implmeneted as extra source terms. Therefore what is solved is:
#    k-eq:  0 = Pk -  rho e + ddy[(mu+mut/sigma_k) dkdy] + Source
#    e-eq:  0 = C_e1 f1 e/k Pk - C_e2 f2 e^2/k + ddy[(mu+mut/sigma_e)dedy] 
#               + Source
#
# Input:
#   u           velocity
#   k           turbulent kinetic energy, from previous time step
#   e           turbulent kinetic energy dissipation rate per unit volume,  
#               from previous time step
#   r           density
#   mu          molecular viscosity
#   ReT         friction Reynolds number ReT=utau r_wall h/ mu_wall
#   mesh        mesh structure
#   compFlag    flag to solve the model with compressible modifications
#
# Output:
#   mut         eddy viscosity or turbulent viscosity
#   k           solved turbulent kinetic energy
#   e           solved turbulent kinetic energy dissipation rate per unit
#               volume

# ------------------------------------------------------------------------
#                  Basic Libraries
# ------------------------------------------------------------------------

from  os.path  import  abspath, dirname
from  sys      import  path              as  sys_path

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(dirname(abspath(__file__))))
from  cfd_solver  import  CFD_solver
sys_path.pop()

# ------------------------------------------------------------------------
#                  Utility functions
# ------------------------------------------------------------------------

found_tag  =  lambda tag, args:  (tag in args) and (not (args[tag] is None))
power      =  lambda a  , b   :  a**b

# ------------------------------------------------------------------------
#                  MK turbulence model
# ------------------------------------------------------------------------

class CFD_Solver_MK_model(CFD_solver):
    def __init__(self, args):
        super().__init__(args)
        
        self.export_vars = ['rans_u'        ,
                            'rans_T'        ,
                            'rans_mu_turb'  ,
                            'rans_MK_k'     ,
                            'rans_MK_e'     ,
                            'beta_distrib_K',
                            'beta_distrib_E',
                            'delta_inject_K',
                            'delta_inject_E']
        
        args                 =  self.args
        torch                =  args['torch']
        self.Ret             =  args['Ret']
        self.tag_rans_model  =  'MK'

        ones_           =  torch.ones_like (self.y)
        zeros_          =  torch.zeros_like(self.y)
        self.rans_MK_k  =  0.1  *ones_
        self.rans_MK_e  =  0.001*ones_

        self.__torch_one    = ones_[0] + 0.

        self.rans_MK_C_mu   = 0.09 
        self.rans_MK_sig_k  = 1.4 
        self.rans_MK_sig_e  = 1.3 
        self.rans_MK_Ce1    = 1.4 
        self.rans_MK_Ce2    = 1.8

        # compressibility correction of (Otero et al. 2018)
        self.compressible_correction  =  args['compressible_correction'] 
        assert not self.compressible_correction                          # not enabled for this study

        n_found     =   0
        for tag in ['beta_distrib_K' ,
                    'beta_distrib_E' ,
                    'delta_inject_K',
                    'delta_inject_E']:
            setattr(self, f'loaded_{tag}',                              found_tag(tag, args))
            setattr(self, tag            , self.as_tensor(args[tag]) if found_tag(tag, args) else {'delta': zeros_+0., 'beta_': ones_+0.}[tag[:5]])
            n_found  +=                                                 found_tag(tag, args)
        assert n_found <= 1

        # Relaxation factors
        if 'jimenez_re4200' in args['case'].lower():
            self.underrelax_K  = 0.7
            self.underrelax_E  = 0.7
        else:
            self.underrelax_K  = 0.9
            self.underrelax_E  = 0.9
    
    def turb_model(self):
        args   =  self.args
        torch  =  args['torch']

        d      =  self.y
        r      =  self.rans_rho_molec
        mu     =  self.rans_mu_molec
        
        yplus  =  d*torch.sqrt(r/r[0])/(mu/mu[0])*self.Ret
        
        u      =  self.rans_u
        k      =  self.rans_MK_k
        e      =  self.rans_MK_e

        # model constants
        C_mu   =  self.rans_MK_C_mu  
        sig_k  =  self.rans_MK_sig_k
        sig_e  =  self.rans_MK_sig_e
        Ce1    =  self.rans_MK_Ce1  
        Ce2    =  self.rans_MK_Ce2  
        
        # wall boundary conditions
        e_BC = mu[0]/r[0]*k[1]/power(d[ 1], 2)

        # model functions 
        ReTurb    = r*power(k, 2)/(mu*e)
        f2        = (1.-2./9.*torch.exp(-power(ReTurb/6., 2)))*power(1-torch.exp(-yplus/5.), 2)
        ReTurb[0] = 1e-8
        fmue      = (1-torch.exp(-yplus/70.))*(1.0+3.45/power(ReTurb, 0.5))
        fmue[0]   = 0.0
        
        # eddy viscosity
        self.rans_mu_turb  =  mut  =  C_mu*fmue*r/e*power(k,2)
        one                        =  self.__torch_one

        mut[1:] = torch.minimum(torch.maximum(mut[1:],1e-10*one),100*one)

        # turbulent production: Pk = mut*dudy^2
        Pk = mut * self.get_grady(u)**2

        # ---------------------------------------------------------------------
        #         k-equation
        # ---------------------------------------------------------------------

        # effective viscosity
        if self.compressible_correction == 1:
            raise Exception('Compressibility correction should not be enabled!')
            mueff      =  (mu + mut/sig_k)/torch.sqrt(r)
            fs         =  r
            fd         =  1/torch.sqrt(r)
        else:
            mueff      =  mu + mut/sig_k
            fs  =  fd  =  1.0
        
        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
        A = self.implicit_diffusivity_matrix(mueff, fd=fd, substract_diag = r*e/k/fs*self.beta_distrib_K)
        
        # Right-hand-side
        b  = -Pk - self.delta_inject_K

        # Wall boundary conditions
        k[0] = 0 # Restore k_BC

        # Solve TKE
        self.rans_MK_k  =  k  =  self.solve_eq(k*fs, A, b, self.underrelax_K)/fs
        k[1:] = torch.maximum(k[1:], 1e-12*one)

        # ---------------------------------------------------------------------
        #         e-equation
        # ---------------------------------------------------------------------

        # effective viscosity
        if self.compressible_correction == 1:
            raise Exception('Compressibility correction should not be enabled!')
            mueff  =  (mu + mut/sig_e)/torch.sqrt(r)
            fs     =  power(r, 1.5)
            fd     =  1/r
        else:
            mueff      =  mu + mut/sig_e
            fs  =  fd  =  1.0
        
        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
        A = self.implicit_diffusivity_matrix(mueff, fd=fd, substract_diag = Ce2*f2*r*e/k/fs*self.beta_distrib_E)
        
        k[0] = 1e-8 # avoid division by zero in the next line

        # Right-hand-side
        b = -e/k*Ce1*Pk - self.delta_inject_E

        k[0]   =  0    # restore k_BC
        e[0]   =  e_BC # wall boundary conditions

        # Solve eps equation
        self.rans_MK_e  =  e  =  self.solve_eq(e*fs, A, b, self.underrelax_E)/fs
        e[1:]  =  torch.maximum(e[1:], 1e-12*one)

    def calc_budgets_MK(self, return_Dest_E = False, full_budgets = False):
        torch = self.args['torch']

        # Calculate the residual of U, k and e
        #    u-eq: RU = ddy[(mu+mut) dudy] + 1
        #    k-eq: RK = Pk -  rho e + ddy[(mu+mut/sigma_k) dkdy]
        #    e-eq: RE = C_e1 f1 e/k Pk - r C_e2 f2 e^2/k + ddy[(mu+mut/sigma_e)dedy] 
        
        Ce2    =  self.rans_MK_Ce2   
        
        mu     =  self.rans_mu_molec
        r      =  self.rans_rho_molec
        
        k      =  torch.clone(self.rans_MK_k)
        e      =              self.rans_MK_e

        k[0]   = 1e-12


        # model functions 
        d         =  self.y
        yplus     =  d*torch.sqrt(r/r[0])/(mu/mu[0])*self.Ret
        ReTurb    = r*power(k, 2)/(mu*e)
        f2        = (1.-2./9.*torch.exp(-power(ReTurb/6., 2)))*power(1-torch.exp(-yplus/5.), 2)

        # early calculation of Dest_E to return value efficiently
        Dest_E    =  -r*Ce2*f2*(e**2)/k
        Dest_E[0] =  0.
        
        if return_Dest_E:
            return Dest_E

        C_mu   =  self.rans_MK_C_mu  
        sig_k  =  self.rans_MK_sig_k 
        sig_e  =  self.rans_MK_sig_e 
        Ce1    =  self.rans_MK_Ce1   

        ReTurb[0] = 1e-8
        fmue      = (1-torch.exp(-yplus/70.))*(1.0+3.45/power(ReTurb, 0.5))
        fmue[0]   = 0.0
        
        mut       =  C_mu*fmue*r/e*(k**2)
        # mueff_u =  mu + mut
        mueff_k   =  mu + mut/sig_k
        mueff_e   =  mu + mut/sig_e
        
        grad          = self.get_grady
        nabla         = self.get_concy
        get_diffusion = lambda mueff, var: grad(mueff) * grad(var) + mueff * nabla(var)

        Pk            =  mut*(grad(self.rans_u)**2)

        Prod_K        =   Pk
        Dest_K        =  -r*e
        Diff_K        =   get_diffusion(mueff_k, k)

        Prod_E        =  Ce1*e/k*Pk
        Diff_E        =  get_diffusion(mueff_e, e)

        # RU  =                                     get_diffusion(mueff_u, u) + 1
        # RK  =  Pk         - r*e                 + get_diffusion(mueff_k, k)
        # RE  =  Ce1*e/k*Pk - r*Ce2*f2*(e**2)/k + get_diffusion(mueff_e, e)
        # 
        # #    u-eq: RU = ddy[(mu+mut) dudy] + 1
        # #    k-eq: RK = Pk -  rho e + ddy[(mu+mut/sigma_k) dkdy]
        # #    e-eq: RE = Ce1 f1 e/k Pk - r C_e2 f2 e^2/k + ddy[(mu+mut/sigma_e)dedy] 
        # 
        # RU[0] = 0.
        # RK[0] = 0.
        # RE[0] = 0.

        Prod_K[0] = Dest_K[0] = Diff_K[0] = Prod_E[0] = Dest_E[0] = Diff_E[0] = 0.

        if full_budgets:
            return { 'Prod_K':  Prod_K, 'Dest_K':  Dest_K, 'Diff_K':  Diff_K, 'extra_delta_K': Dest_K*(self.beta_distrib_K - 1), 'beta_K': self.beta_distrib_K,
                     'Prod_E':  Prod_E, 'Dest_E':  Dest_E, 'Diff_E':  Diff_E, 'extra_delta_E': Dest_E*(self.beta_distrib_E - 1), 'beta_E': self.beta_distrib_E,
                   }
        
        # R_all = self.as_tensor( RU.cpu().detach().numpy().tolist() + 
        #                         RK.cpu().detach().numpy().tolist() + 
        #                         RE.cpu().detach().numpy().tolist() )
        # 
        # rans_MK_Rn     = R_all
        # rans_MK_Prod_K = Prod_K
        # rans_MK_Dest_K = Dest_K
        # rans_MK_Diff_K = Diff_K
        # 
        # rans_MK_Prod_E = Prod_E
        # rans_MK_Dest_E = Dest_E
        # rans_MK_Diff_E = Diff_E
        
        # Dest_K[0]  =  1e-12 # avoid division by zero
        # Dest_E[0]  =  1e-12

        # rans_MK_Beta_K_calc  =  -(Prod_K + Diff_K)/Dest_K
        # rans_MK_Beta_E_calc  =  -(Prod_E + Diff_E)/Dest_E
        # 
        # rans_MK_Beta_K_calc[0] = 1.
        # rans_MK_Beta_E_calc[0] = 1.

        get_basis = lambda all_a: max(float(a.abs().max().item()) for a in all_a)
        
        return {'Rk_basis': get_basis([Prod_K, Dest_K, Diff_K]),
                'Re_basis': get_basis([Prod_E, Dest_E, Diff_E])}


# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    sys_path.append(dirname(dirname(abspath(__file__))))
    from  cfd_solver  import  quick_benchmark
    sys_path.pop()

    quick_benchmark(CFD_Solver_MK_model)



