#**************************************************************************
#       Implementation of SA model (Spalart-Allmaras 1992 AIAA)
#       Reference,
#       Spalart, A. and Allmaras, S., "One equation turbulence model for 
#       aerodynamic flows", Recherche Aerospatiale-French edition, 1994.
#**************************************************************************
# The SA model consists of a transport equation for an eddy viscosity-like
# scalar derived using dimensional analysis, Galilean invariance and
# empiricism.
#
# Conventional models without compressible modifications:
#    nuSA-eq:  0 = cb1 Shat nuSA - cw1 fw (nuSA/wallDist)^2
#                  + 1/cb3 ddy[(nu+nuSA) dnuSAdy] + cb2/cb3 (dnuSAdy)^2
#
# Otero et.al model:
#    nuSA-eq:  0 = cb1 Shat nuSA - cw1 fw (nuSA/wallDist)^2
#                  + 1/cb3 1/rho ddy[rho (nu+nuSA) dnuSAdy]
#                  + 1/cb3 1/rho ddy[nuSA/2 (nu+nuSA) drhody]
#                  + cb2/cb3 1/rho (d(sqrt(rho) nuSA)dy)^2
#
# Catris, S. and Aupoix, B., "Density corrections for turbulence
#       models", Aerosp. Sci. Techn., 2000.
#    nuSA-eq:  0 = cb1 Shat nuSA - cw1 fw (nuSA/wallDist)^2
#                  + 1/cb3 1/rho ddy[rho (nu+nuSA) dnuSAdy]
#                  + 1/cb3 1/rho ddy[nuSA/2 (nuSA) drhody]
#                  + cb2/cb3 1/rho (d(sqrt(rho) nuSA)dy)^2
#
# For simplicty, the extra density factors of the Otero et.al and Catris/Aupoix
# models were implmeneted as extra source terms. Therefore what is solved is:
# Conventional model:
#    nuSA-eq:  0 = cb1 Shat nuSA - cw1 fw (nuSA/wallDist)^2
#                  + 1/cb3 ddy[(nu+nuSA) dnuSAdy] + cb2/cb3 (dnuSAdy)^2
#                  + Source
#
# Input:
#   u           velocity
#   nuSA        eddy viscosity-like scalar, from previous time step
#   r           density
#   mu          molecular viscosity
#   self        self structure
#   compFlag    flag to solve the model with compressible modifications
#
# Output:
#   mut         eddy viscosity or turbulent viscosity
#   nuSA        solved eddy viscosity-like scalar

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
#                  SA turbulence model
# ------------------------------------------------------------------------

class CFD_Solver_SA(CFD_solver):
    def __init__(self, args):
        super().__init__(args)
        assert hasattr(self, 'args')
        self.Ret  =  args['Ret']
        self.tag_rans_model = 'SA'

        # compressibility correction of (Otero et al. 2018)
        assert not args['compressible_correction']  # not enabled for this study

        self.rans_nuSA = self.rans_mu_molec / self.rans_rho_molec

        self.underrelax_SA = 0.75
        
        torch = self.args['torch']
        zero_         =  self.y[0]*0.
        self.maximum  =  lambda a,b: torch.maximum(a+zero_, b+zero_)
        self.minimum  =  lambda a,b: torch.minimum(a+zero_, b+zero_)

    def turb_model(self):
        
        n     =  self.n
        u     =  self.rans_u
        r     =  self.rans_rho_molec
        mu    =  self.rans_mu_molec
        nuSA  =  self.rans_nuSA

        torch = self.args['torch']
        zeros = torch.zeros
        ones  = torch.ones

        maximum  =  self.maximum
        minimum  =  self.minimum

        # Model constants
        cv1_3   = power(7.1, 3.0)
        cb1     = 0.1355
        cb2     = 0.622
        cb3     = 2.0/3.0
        inv_cb3 = 1.0/cb3
        kappa_2 = power(0.41, 2.0)
        cw1     = cb1/kappa_2 + (1.0+cb2)/cb3
        cw2     = 0.3
        cw3_6   = power(2.0, 6.0)

        # Model functions
        strMag    =  self.get_grady(u).abs() # VortRate = StrainRate in fully developed channel
        wallDist  =  maximum(self.y, 1e-8)

        inv_wallDist2 = 1/power(wallDist, 2)

        chi           = nuSA*r/mu
        fv1           = power(chi, 3)/(power(chi, 3) + cv1_3)
        fv2           = 1.0 - (chi/(1.0+(chi*fv1)))
        inv_kappa2_d2 = inv_wallDist2*(1.0/kappa_2)
        Shat          = strMag + inv_kappa2_d2*fv2*nuSA
        inv_Shat      = 1.0/Shat

        r_SA          = minimum(nuSA*inv_kappa2_d2*inv_Shat, 10.0)
        g             = r_SA + cw2*(power(r_SA, 6) - r_SA)
        g_6           = power(g, 6)
        fw_           = power(((1.0 + cw3_6)/(g_6+ cw3_6)), (1/6))
        fw            = g*fw_

        # Eddy viscosity
        mut       = zeros(n)
        mut[1:] = fv1[1:]* nuSA[1:]*r[1:]
        mut[1:] = minimum(maximum(mut[1:], 0.0), 100.0)

        if self.args['compressible_correction']:  
            raise Exception('Incompressibility correction should not be enabled (for this paper)')
            nueff = (mu/r + nuSA)*r
            fs    = torch.sqrt(r)
            fd    = 1/r
            drdy  = self.get_grady(r)
            Di    = nueff*nuSA*drdy
            drho  = 0.5*inv_cb3/r*(self.get_grady(Di))
        else:
            nueff = (mu/r + nuSA)
            fs    = ones(n)
            fd    = ones(n)
            drho  = zeros(n)

        # ---------------------------------------------------------------------
        # nuSA-equation 

        # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
        A = self.implicit_diffusivity_matrix(inv_cb3*nueff, fd=fd, substract_diag = cw1*fw*nuSA*inv_wallDist2)

        # Right hand side
        dnudy = self.get_grady(fs*nuSA)
        b     = - cb1*Shat*nuSA - cb2*inv_cb3*power(dnudy, 2) - drho

        # Wall boundary conditions
        nuSA[0] = 0.0

        # Solve
        nuSA      =  self.solve_eq(nuSA, A, b, self.underrelax_SA)
        nuSA[1:]  =  maximum(nuSA[1:], 1.e-12)
        self.rans_mu_turb  =  mut
        self.rans_nuSA     =  nuSA

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    sys_path.append(dirname(dirname(abspath(__file__))))
    from  cfd_solver  import  quick_benchmark
    sys_path.pop()

    quick_benchmark(CFD_Solver_SA)