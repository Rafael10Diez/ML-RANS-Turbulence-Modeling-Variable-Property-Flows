#**************************************************************************
#       Implementation of the v2f model
#       Reference,
#       Medic, G. and Durbin, P.A., "Towards improved prediction of heat 
#       transfer on turbine blades", ASME, J. Turbomach. 2012.
#**************************************************************************
#
# Conventional models without compressible modifications:
#    k-eq:  0 = Pk - rho e + ddy[(mu+mut/sigma_k) dkdy]
#    e-eq:  0 = (C_e1 Pk - C_e2 rho e)/T  + ddy[(mu+mut/sigma_e)dedy] 
#    v2-eq: 0 = rho k f - 6 rho v2 e/k + ddy[(mu+mut/sigma_k) dv2dy]
#    f-eq:  L^2 d2fdy2 - f = [C1 -6v2/k -2/3(C1-1)]/T -C2 Pk/(rho k)  
#                
# Otero et.al compressibility modifications:
#    k-eq:  0 = Pk - rho e
#               + 1/sqrt(rho) ddy[1/sqrt(rho) (mu+mut/sigma_k) d(rho k)dy]
#    e-eq:  0 = (C_e1 Pk - C_e2 rho e)/T 
#               + 1/rho ddy[1/sqrt(rho) (mu+mut/sigma_e) d(rho^1.5 e)dy] 
#    v2-eq: 0 = rho k f - 6 rho v2 e/k 
#               + ddy[1/sqrt(rho) (mu+mut/sigma_k) d(rho v2)dy]
#    f-eq:  L^2 d2fdy2 - f = [C1 -6v2/k -2/3(C1-1)]/T -C2 Pk/(rho k) 
#
# Catris, S. and Aupoix, B., "Density corrections for turbulence
#       models", Aerosp. Sci. Techn., 2000.
#    k-eq:  0 = Pk - rho e 
#               + ddy[1/rho (mu+mut/sigma_k) d(rho k)dy]
#    e-eq:  0 = (C_e1 Pk - C_e2 rho e)/T 
#               + 1/rho ddy[1/sqrt(rho) (mu+mut/sigma_e) d(rho^1.5 e)dy]
#    v2-eq: 0 = rho k f - 6 rho v2 e/k 
#               + ddy[1/rho (mu+mut/sigma_k) d(rho v2)dy]
#    f-eq:  L^2 d2fdy2 - f = [C1 -6v2/k -2/3(C1-1)]/T -C2 Pk/(rho k) 
#
# Input:
#   u           velocity
#   k           turbulent kinetic energy, from previous time step
#   e           turbulent kinetic energy dissipation rate per unit volume,  
#               from previous time step
#   v2          wall normal velocity fluctuation, from previos time step
#   r           density
#   mu          molecular viscosity
#   self        self structure
#   compFlag    flag to solve the model with compressible modifications
#
# Output:
#   mut         eddy viscosity  (turbulent viscosity)
#   k           turbulent kinetic energy
#   e           turbulent kinetic energy dissipation rate
#   v2          wall normal velocity fluctuation

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
#                  V2F turbulence model
# ------------------------------------------------------------------------

class CFD_Solver_V2F(CFD_solver):
    def __init__(self, args):
        super().__init__(args)
        assert hasattr(self, 'args')
        self.Ret  =  args['Ret']
        self.tag_rans_model  =  'V2F'

        # compressibility correction of (Otero et al. 2018)
        assert not args['compressible_correction']  # not enabled for this study
        
        torch  =  self.args['torch']
        self.rans_V2F_k   =  0.1  *torch.ones_like(self.y)
        self.rans_V2F_e   =  0.001*torch.ones_like(self.y)
        self.rans_V2F_v2  =  self.rans_V2F_k/3.

        # self.underrelax_T = 0.95
        zero_         =  self.y[0]*0.
        self.maximum  =  lambda a,b: torch.maximum(a+zero_, b+zero_)
        self.minimum  =  lambda a,b: torch.minimum(a+zero_, b+zero_)

    def turb_model(self):

        torch    =  self.args['torch']
        zeros    =  torch.zeros
        ones     =  torch.ones
        sqrt     =  torch.sqrt
        maximum  =  self.maximum
        minimum  =  self.minimum

        n      =  self.n
        f      =  zeros(n)
        y      =  self.y
        r      =  self.rans_rho_molec
        mu     =  self.rans_mu_molec
        u      =  self.rans_u

        k  = self.rans_V2F_k 
        e  = self.rans_V2F_e 
        v2 = self.rans_V2F_v2

        # Model constants
        cmu  = 0.22 
        sigk = 1.0 
        sige = 1.3 
        Ce2  = 1.9
        Ct   = 6 
        Cl   = 0.23 
        Ceta = 70 
        C1   = 1.4 
        C2   = 0.3

        # Relaxation factors
        underrelaxK  = 0.8
        underrelaxE  = 0.8
        underrelaxV2 = 0.8
        # Time and length scales, eddy viscosity and turbulent production
        Tt  = maximum(k/e, Ct*power(mu/(r*e), 0.5))
        Lt  = Cl*maximum(power(k, 1.5)/e, Ceta*power(power(mu/r, 3)/e, 0.25))
        mut = maximum(cmu*r*v2*Tt, 0.0)
        Pk  = mut*power(self.get_grady(u), 2.0)

        v2[ 0 ] = 1e-12
        k [ 0 ] = 1e-12

        # ---------------------------------------------------------------------
        # k-equation

        # effective viscosity
        if self.args['compressible_correction']:
            raise Exception('Incompressibility correction should not be enabled (for this paper)')
            mueff = (mu + mut/sigk)/sqrt(r)
            fs    = r
            fd    = 1/sqrt(r)
        else:
            mueff = mu + mut/sigk
            fs    = ones(n)
            fd    = ones(n)

        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
        A    = self.implicit_diffusivity_matrix(mueff, fd=fd, substract_diag =  r*e/maximum(k,1e-12)/fs)

        # Right-hand-side
        b = -Pk

        # Wall boundary conditions
        k[0] = 0.0

        # Solve
        k = self.solve_eq(k*fs, A, b, underrelaxK)/fs
        k = maximum(k, 1.e-12)

        # ---------------------------------------------------------------------
        # e-equation

        # effective viscosity
        if self.args['compressible_correction']:
            raise Exception('Incompressibility correction should not be enabled (for this paper)')
            mueff = (mu + mut/sige)/sqrt(r)
            fs    = power(r, 1.5)
            fd    = 1/r
        else:
            mueff = mu + mut/sige
            fs    = ones(n)
            fd    = ones(n)

        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
        A    = self.implicit_diffusivity_matrix(mueff, fd=fd, substract_diag =  Ce2/Tt*r/fs)

        # Right-hand-side
        Ce1 = 1.4*(1 + 0.045*sqrt(k/v2))
        b = -1/Tt*Ce1*Pk

        # Wall boundary conditions
        e[0 ] = mu[0 ]*k[1 ]/r[0 ]/power(self.y[1 ]-self.y[0 ], 2)

        # Solve
        e = self.solve_eq(e*fs, A, b, underrelaxE)/fs
        e[1:] = maximum(e[1:], 1.e-12)

        # ---------------------------------------------------------------------
        # f-equation 

        # implicitly treated source term
        Lt2_view = (Lt*Lt)
        A        =  {'A'      : self.d2_dy2 * Lt2_view[1:].reshape(-1,1) ,
                     'wall_BC': self.wall_BC_d2_dy2   * Lt2_view[0] }
        A['A'][:,1] -= 1

        # Right-hand-side
        vok  = v2/k
        rhsf = ((C1-6)*vok - 2/3*(C1-1))/Tt - C2*Pk/(r*k)

        # Solve
        f = self.solve_eq(f,A,rhsf,1)
        f[1:] = maximum(f[1:], 1.e-12)

        # ---------------------------------------------------------------------
        # v2-equation: 

        # effective viscosity and pre-factors for compressibility implementation
        if self.args['compressible_correction']:
            raise Exception('Incompressibility correction should not be enabled (for this paper)')
            mueff = (mu + mut)/sqrt(r)
            fs    = r
            fd    = 1/sqrt(r)
        else:
            mueff = mu + mut
            fs    = ones(n)
            fd    = ones(n)

        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
        A    = self.implicit_diffusivity_matrix(mueff, fd=fd, substract_diag =  6.0*r*e/k/fs)

        # Right-hand-side
        b = -r*k*f

        # Wall boundary conditions
        v2[0] = 0.0

        # Solve
        v2 = self.solve_eq(v2*fs, A, b, underrelaxV2)/fs
        v2[1:] = maximum(v2[1:], 1.e-12)
        v2[0] = 1e-12

        k [0] = 0.0
        v2[0] = 0.0

        self.rans_mu_turb = mut
        self.rans_V2F_k   =  k
        self.rans_V2F_e   =  e
        self.rans_V2F_v2  =  v2

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    sys_path.append(dirname(dirname(abspath(__file__))))
    from  cfd_solver  import  quick_benchmark
    sys_path.pop()

    quick_benchmark(CFD_Solver_V2F)