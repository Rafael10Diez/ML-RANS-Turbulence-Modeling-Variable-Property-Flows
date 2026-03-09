#**************************************************************************
#       Implementation of k-omega SST
#       Reference,
#       Menter, F.R., "Zonal Two equation k-omega turbulence models for 
#       aerodynamic flows", AIAA 93-2906, 1993.
#**************************************************************************
# Two equation turbulence model, which combines Wilcox k-omega model
# and k-epsilon mode through a blending funcion
#
# Conventional models without compressible modifications:
#    k-eq:  0 = Pk - (beta_star rho k om) + ddy[(mu+mut*sigma_k) dkdy]
#    om-eq: 0 = alpha rho/mut Pk - (betar rho om^2) 
#               + ddy[(mu+mut*sigma_om)domdy] + (1-BF1) CDkom
#
# Otero et.al model:
#    k-eq:  0 = Pk - (beta_star rho k om) 
#               + 1/sqrt(rho) ddy[1/sqrt(rho) (mu+mut*sigma_k) d(rho k)dy]
#    om-eq: 0 = alpha rho/mut Pk - (betar rho om^2) 
#               + ddy[1/sqrt(rho) (mu+mut*sigma_om)d(sqrt(rho) om)dy] 
#               + (1-BF1) CDkom_mod
#
# Catris, S. and Aupoix, B., "Density corrections for turbulence
#       models", Aerosp. Sci. Techn., 2000.
#    k-eq:  0 = Pk - (beta_star rho k om) 
#               + ddy[1/rho (mu+mut*sigma_k) d(rho k)dy]
#    om-eq: 0 = alpha rho/mut Pk - (betar rho om^2) 
#               + ddy[1/sqrt(rho) (mu+mut*sigma_om)d(sqrt(rho) om)dy] 
#               + (1-BF1) CDkom
#
# For simplicty, the extra density factors of the Otero et.al and Catris/Aupoix  
# models were implmeneted as extra source terms. Therefore what is solved is:
#    k-eq:  0 = Pk - (beta_star rho k om) + ddy[(mu+mut*sigma_k) dkdy]
#               + Source
#    om-eq: 0 = alpha rho/mut Pk - (betar rho om^2) 
#               + ddy[(mu+mut*sigma_om)domdy] + (1-BF1) CDkom + Source
#
# Input:
#   u           velocity
#   k           turbulent kinetic energy, from previous time step
#   om          turbulent kinetic energy dissipation rate, from previous 
#               time step
#   r           density
#   mu          molecular viscosity
#   self        self structure
#   compFlag    flag to solve the model with compressible modifications
#
# Output:
#   mut         eddy viscosity or turbulent viscosity
#   k           turbulent kinetic energy
#   om          turbulent kinetic energy dissipation rate

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
#                  SST turbulence model
# ------------------------------------------------------------------------

class CFD_Solver_SST(CFD_solver):
    def __init__(self, args):
        super().__init__(args)
        assert hasattr(self, 'args')
        self.Ret  =  args['Ret']
        self.tag_rans_model  =  'SST'

        # compressibility correction of (Otero et al. 2018)
        assert not args['compressible_correction']  # not enabled for this study
        
        torch             =  self.args['torch']
        self.rans_SST_k   =  0.1      *torch.ones_like(self.y)
        self.rans_SST_om  =  0.001/0.1*torch.ones_like(self.y)

        self.underrelax_T = 0.95
        
        zero_         =  self.y[0]*0.
        self.maximum  =  lambda a,b: torch.maximum(a+zero_, b+zero_)
        self.minimum  =  lambda a,b: torch.minimum(a+zero_, b+zero_)

    def turb_model(self):

        torch    =  self.args['torch']
        ones     =  torch.ones
        sqrt     =  torch.sqrt
        tanh     =  torch.tanh

        maximum  =  self.maximum
        minimum  =  self.minimum

        n        =  self.n
        r        =  self.rans_rho_molec
        mu       =  self.rans_mu_molec
        u        =  self.rans_u
        
        k        =  self.rans_SST_k
        om       =  self.rans_SST_om

        # model constants
        sigma_k1  = 0.85
        sigma_k2  = 1.0
        sigma_om1 = 0.5
        sigma_om2 = 0.856
        beta_1    = 0.075
        beta_2    = 0.0828
        betaStar  = 0.09
        a1        = 0.31
        alfa_1    = beta_1/betaStar - sigma_om1*0.41**2.0/betaStar**0.5
        alfa_2    = beta_2/betaStar - sigma_om2*0.41**2.0/betaStar**0.5    

        # Relaxation factors
        underrelaxK  = 0.6
        underrelaxOm = 0.4

        # required gradients
        dkdy  = self.get_grady(k)
        domdy = self.get_grady(om)

        wallDist = maximum(self.y, 1.0e-8)


        # VortRate = StrainRate in fully developed channel
        strMag = self.get_grady(u).abs()

        # Blending functions 
        CDkom  = 2.0*sigma_om2*r/om*dkdy*domdy
        gamma1 = 500.0*mu/(r*om*wallDist*wallDist)
        gamma2 = 4.0*sigma_om2*r*k/(wallDist*wallDist*maximum(CDkom,1.0e-20))
        gamma3 = sqrt(k)/(betaStar*om*wallDist)
        gamma  = minimum(maximum(gamma1, gamma3), gamma2)
        bF1    = tanh(power(gamma, 4.0))
        gamma  = maximum(2.0*gamma3, gamma1)
        bF2    = tanh(power(gamma, 2.0))

        # more model constants
        alfa     = alfa_1*bF1    + (1-bF1)*alfa_2
        beta     = beta_1*bF1    + (1-bF1)*beta_2
        sigma_k  = sigma_k1*bF1  + (1-bF1)*sigma_k2
        sigma_om = sigma_om1*bF1 + (1-bF1)*sigma_om2

        # Eddy viscosity
        om    [om    .abs()<1e-14] = 1e-14
        strMag[strMag.abs()<1e-14] = 1e-14
        bF2   [bF2   .abs()<1e-14] = 1e-14
        zeta = minimum(1.0/om, a1/(strMag*bF2))
        mut = r*k*zeta
        mut = minimum(maximum(mut,0.0),100.0)

        # ---------------------------------------------------------------------
        # om-equation
        # ---------------------------------------------------------------------

        # effective viscosity
        if self.args['compressible_correction']:
            raise Exception('Incompressibility correction should not be enabled (for this paper)')
            mueff = (mu + sigma_om*mut)/sqrt(r)
            fs    = sqrt(r)
        else:
            mueff = mu + sigma_om*mut
            fs    = ones(n)

        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
        A  =  self.implicit_diffusivity_matrix(mueff, fd=1, substract_diag =  beta*r*om/fs)

        # Right-hand-side
        b = -alfa*r*strMag*strMag - (1-bF1)*CDkom

        # Wall boundary conditions
        om[0 ] = 60.0*mu[0 ]/beta_1/r[0 ]/wallDist[1 ]/wallDist[1 ]

        # Solve
        om     = self.solve_eq(om*fs, A, b, underrelaxOm)/fs
        om[1:] = maximum(om[1:], 1.e-12)
        
        # ---------------------------------------------------------------------
        # k-equation    
        # ---------------------------------------------------------------------

        # effective viscosity
        if self.args['compressible_correction']:
            raise Exception('Incompressibility correction should not be enabled (for this paper)')
            mueff = (mu + sigma_k*mut)/sqrt(r)
            fs    = r
            fd    = sqrt(r)
        else:
            mueff = mu + sigma_k*mut
            fs    = ones(n)
            fd    = ones(n)

        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy

        A  =  self.implicit_diffusivity_matrix(mueff, fd=fd, substract_diag = betaStar*r*om/fs)

        # Right-hand-side
        Pk = minimum(mut*strMag*strMag, 20*betaStar*k*r*om)
        b  = -Pk

        # Wall boundary conditions
        k[0] = 0.0

        # Solve
        k = self.solve_eq(k*fs, A, b, underrelaxK)/fs
        k[1:] = maximum(k[1:], 1.e-12)

        self.rans_mu_turb  =  mut
        self.rans_SST_k    =  k
        self.rans_SST_om   =  om

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    sys_path.append(dirname(dirname(abspath(__file__))))
    from  cfd_solver  import  quick_benchmark
    sys_path.pop()

    quick_benchmark(CFD_Solver_SST)