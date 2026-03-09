# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from  time                 import  time
from  scipy.linalg.lapack  import  dgtsv          as  tridiag_solver

# ------------------------------------------------------------------------
#                  Compute y*, u*
# ------------------------------------------------------------------------

def get_ystar_ustar(D):
    # https://pure.tudelft.nl/ws/portalfiles/portal/19409280/Patel_dissertation.pdf
    torch   =  D['torch']
    rho     =  D['rho']
    mu      =  D['mu' ]
    u       =  D['u'  ]
    Ret     =  D['Ret']
    y       =  D['y']
    
    rho_w   =  rho[0]
    mu_w    =  mu [0]
    Re_sls  =  Ret * torch.sqrt(rho/rho_w)*mu_w/mu
    
    ustar   =  torch.zeros_like(u)
    
    for i in range(1,int(u.numel())):
        du             =       u     [i] - u     [i-1]
        rho_mid        =  0.5*(rho   [i] + rho   [i-1])
        y_mid          =  0.5*(y     [i] + y     [i-1])
        Ret_star       =  0.5*(Re_sls[i] + Re_sls[i-1])
        grad_Ret_star  =      (Re_sls[i] - Re_sls[i-1])/(y[i]- y[i-1])
        d_uvd          = torch.sqrt(rho_mid/rho_w)*du                  # (4.3)
        d_ustar        = (1.+y_mid/Ret_star*grad_Ret_star)*d_uvd       # (4.8)
        ustar[i]       = ustar[i-1] + d_ustar
    
    return dict(ustar = ustar   ,
                ystar = y*Re_sls)

def resample(y,n):
    print('Resampler activated!')
    import  numpy              as      np
    from    scipy.interpolate  import  interp1d
    def adim(n):
        A = np.array(range(n)).astype(float)
        return A/A.max()
    return interp1d(adim(len(y)),y)(adim(n))

# ------------------------------------------------------------------------
#                  CFD Solver
# ------------------------------------------------------------------------

class CFD_solver:
    def __init__(self, args):
        if ('n' in args) and args['n'] != len(args['y_grid']):
            raise Exception(f"Mismatch in grid parameters (n = {args['n']}) (len(y_grid) = {len(args['y_grid'])})")
            args['y_grid'] = resample(args['y_grid'], args['n'])
        self.args        =  args
        self.as_tensor   =  lambda x: self.args['torch'].tensor(x                                  ,
                                                                dtype  = self.args['torch'].double ,
                                                                device = self.args['tensor_device'])

        torch             =  self.args['torch']
        self.y            =  self.as_tensor(self.args['y_grid'])
        self.n            =  int(self.y.numel())
        self.underrelax_T = 1
        
        # Boundary Condition is assumed at the wall
        self.__mk_ddy_d2dy2()

        self.rans_u    =  torch.zeros_like(self.y)
        self.cp        =  self.args['cp']
        self.Pr_turb   =  self.args['Pr_turb']

        # assert self.args['bool_solve_energy']

        if self.args['bool_solve_energy']:
            self.rans_T  =  torch.ones_like(self.y)
        else:
            self.rans_T  =  self.as_tensor(self.args['ref_dns']['T_dns'])
        
        self.update_properties()

    def solve_u(self):
        torch       = self.args['torch']
        mu_eff      =  self.rans_mu_molec + self.rans_mu_turb
        A           =  self.implicit_diffusivity_matrix(mu_eff)
        b           = -torch.ones_like(self.y) # notice minus sign
        self.rans_u = self.solve_eq(self.rans_u, A, b)
    
    def solve_T(self):
        torch  =                             self.args['torch']
        b_rhs  = - torch.ones_like(self.y) * self.args['constant_Sq_heat']# heating has negative term at rhs
        if self.args['use_visc_heating']:
            
            dudy      =  self.get_grady(self.rans_u)
            tau_y     =  1-self.y    # because tau_wall = u_star=rho_w=1 in all cases considered
            b_rhs    -=  dudy*tau_y  # viscous heating expression (add with negative sign to rhs)
        
        k_turb       =  self.cp*self.rans_mu_turb / self.Pr_turb
        k_eff        =  self.rans_k_molec + k_turb
        A            =  self.implicit_diffusivity_matrix(k_eff)
        self.rans_T  =  self.solve_eq(self.rans_T, A, b_rhs, self.underrelax_T)
    
    def update_properties(self):
        self.rans_rho_molec  =  self.args['A_sca_rho'] * (self.rans_T**self.args['b_exp_rho'] ) # density
        self.rans_mu_molec   =  self.args['A_sca_mu']  * (self.rans_T**self.args['b_exp_mu']  ) # viscosity
        self.rans_k_molec    =  self.args['A_sca_k']   * (self.rans_T**self.args['b_exp_k']   ) # thermal cond.
    
    def implicit_diffusivity_matrix(self, val, fd=1, substract_diag = None):
        # dval_dy * du_dy + val * d2u_dy2
        grad_val  =  (fd*self.get_grady(val))[1:]
        val       =  (fd*val                )[1:]

        multiply  =  lambda a,b: a * b.view(-1,1)
        
        result  =  multiply(self.d_dy, grad_val) + multiply(self.d2_dy2, val)
        
        if not (substract_diag is None):
            result[:,1]  -=  substract_diag[1:]  # [:,1] is the center diagonal
        
        return result
    
    def get_grady(self, val):
        return sum(self.geom_vars[f'cd1_{p}']*val[self.geom_vars[f'i_{p}']] for p in 'scn')
    
    def get_concy(self, val):
        return sum(self.geom_vars[f'cd2_{p}']*val[self.geom_vars[f'i_{p}']] for p in 'scn')
    
    def solve_eq(self, x, A, b, omega_under_relax = 1):
        
        b     =  b[1:] + 0.
        b[0] -=  x[0] * A[0,0]

        if omega_under_relax < 1:
            A_diag  = A[:,1]
            b      += (1./omega_under_relax - 1) * A_diag * x[1:]
            A[:,1] += (1./omega_under_relax - 1) * A_diag
        
        x    =   x.numpy()
        A     =  A.numpy()
        b     =  b.numpy()
        x[1:] =  tridiag_solver(A[1:  , 0],
                                A[ :  , 1] ,
                                A[ :-1, 2],
                                b       )[-2]
        x     =  self.as_tensor(x)
        return x

    def iterate(self, assert_convergence = True):
        torch       =  self.args['torch']
        base_info   =  lambda iter, residual_u, added='': f'       iter: {iter:9d}{added}  (residual_u: {residual_u:12.6e}) (Elapsed Time = {(time()-t0):.3f}) (iters/sec = {(iter/(time()-t0)):.3f})'
        nmin        =      5
        nmax        =  40_000
        tol_u       =  1e-10
        nprint      =   2_000
        iter        =      0
        residual_u  =  float('inf')
        t0          =  time()
        self.update_properties()
        while (iter <= nmin) or ((residual_u >= tol_u) and (iter <= nmax)):
            u_old = torch.clone(self.rans_u)
            
            if (iter>=1) and self.args['bool_solve_energy']:
                self.solve_T()
                self.update_properties()
            
            self.turb_model()
            self.solve_u()
            
            residual_u  =  float(torch.linalg.norm(self.rans_u - u_old))
            iter       +=  1

            if ((iter % nprint)==0) and self.args['fprint']:
                self.args['fprint'](base_info(iter, residual_u))
        
        self.N_iters_solver  =  iter
        if assert_convergence:
            assert residual_u < tol_u
        
        # self.args['fprint'](base_info(iter, residual_u))
        if self.args['output_last']:
            assert self.args['fprint']
            maxabs     =  lambda x: float(torch.abs(x).max().item())
            ref_u_DNS  =  self.as_tensor(self.args['ref_dns']['u_dns'])
            ref_T_DNS  =  self.as_tensor(self.args['ref_dns']['T_dns'])

            ustar_rans    = get_ystar_ustar({'torch'    : torch              ,
                                             'mu'       : self.rans_mu_molec ,
                                             'rho'      : self.rans_rho_molec,
                                             'u'        : self.rans_u        ,
                                             'Ret'      : self.Ret           ,
                                             'y'        : self.y             })['ustar']
            
            ref_ustar_dns = get_ystar_ustar({'torch'    : torch                                          ,
                                             'mu'       : self.as_tensor(self.args['ref_dns']['mu_dns' ]),
                                             'rho'      : self.as_tensor(self.args['ref_dns']['rho_dns']),
                                             'u'        : ref_u_DNS                                      ,
                                             'Ret'      : self.Ret                                       ,
                                             'y'        : self.y                                         })['ustar']

            self.output_stats = dict( u_rans_tip      =  float(self.rans_u [-1].item())                              ,
                                      T_rans_tip      =  float(self.rans_T [-1].item())                              ,
                                      ustar_rans_tip  =  float(ustar_rans  [-1].item())                              ,
                                      u_dns_tip       =  float(ref_u_DNS    [-1].item())                             ,
                                      T_dns_tip       =  float(ref_T_DNS    [-1].item())                             ,
                                      ustar_dns_tip   =  float(ref_ustar_dns[-1].item())                             ,
                                      u_res           =  maxabs(self.rans_u - ref_u_DNS    ) / maxabs(ref_u_DNS)     ,
                                      T_res           =  maxabs(self.rans_T - ref_T_DNS    ) / maxabs(ref_T_DNS)     ,
                                      ustar_res       =  maxabs(ustar_rans  - ref_ustar_dns) / maxabs(ref_ustar_dns) )
            keys  =  ['u_rans_tip','u_dns_tip','u_res','ustar_rans_tip','ustar_dns_tip','ustar_res','T_rans_tip','T_dns_tip','T_res']
            assert set(keys) == set(self.output_stats.keys())
            self.args['fprint'](base_info(iter, residual_u, added=' (final)') + ' ' + ' '.join(f"({key}: {self.output_stats[key]*(100 if '_res' in key else 1):8.3f}{'%' if '_res' in key else ''})" for key in keys))
    
    def __mk_ddy_d2dy2(self):
        
        coeff_d_dy_south   =  lambda dxn, dxs :  -dxn/(dxs*(dxn + dxs))
        coeff_d_dy_center  =  lambda dxn, dxs :  (dxn - dxs)/(dxn*dxs)
        coeff_d_dy_north   =  lambda dxn, dxs :   dxs/(dxn*(dxn + dxs))
        coeff_d2_dy_south  =  lambda dxn, dxs :   2./(dxs*(dxn + dxs))
        coeff_d2_dy_center =  lambda dxn, dxs :  -2./(dxn*dxs)
        coeff_d2_dy_north  =  lambda dxn, dxs :   2./(dxn*(dxn + dxs))

        assert self.n == self.y.numel()

        zeros_like      =  self.args['torch'].zeros_like
        self.geom_vars  =     {f'{var}_{p}': zeros_like(self.y)        for var in ['cd1','cd2'] for p in 'scn'}
        self.geom_vars.update({f'i_{p}': zeros_like(self.y).long()                              for p in 'scn'})
        
        for ii in range(self.n):
            i_s, i_c, i_n = ii-1, ii, ii+1
            
            if ii == 0:
                i_s = i_n + 1 # use one further point to the north (interpolate from interior)
            
            if ii == (self.n-1):
                i_n, pn = i_s, 's'   # use point from the south (repeated), since the channel is symmetric
                #                      and assign north variable to south position as well
            else:
                pn  = 'n'            # keep north variable intact

            dy_n                             = abs(float( self.y[i_n]- self.y[i_c] )) # must be "abs" due to the symmetry (at i=n)
            dy_s                             =     float( self.y[i_c]- self.y[i_s] )  # must have a sign (at i=1)
            
            self.geom_vars['i_c'  ][ii]       =  i_c
            self.geom_vars['i_n'  ][ii]       =  i_n
            self.geom_vars['i_s'  ][ii]       =  i_s
            
            self.geom_vars[ 'cd1_s'   ][ii]  +=  coeff_d_dy_south  (dy_n,dy_s)
            self.geom_vars[ 'cd1_c'   ][ii]  +=  coeff_d_dy_center (dy_n,dy_s)
            self.geom_vars[f'cd1_{pn}'][ii]  +=  coeff_d_dy_north  (dy_n,dy_s)

            self.geom_vars[ 'cd2_s'   ][ii]  +=  coeff_d2_dy_south (dy_n,dy_s)
            self.geom_vars[ 'cd2_c'   ][ii]  +=  coeff_d2_dy_center(dy_n,dy_s)
            self.geom_vars[f'cd2_{pn}'][ii]  +=  coeff_d2_dy_north (dy_n,dy_s)

        self.d_dy    =  self.__stack_diags('cd1')
        self.d2_dy2  =  self.__stack_diags('cd2')
    
    def export_state(self):
        return { tag: getattr(self, tag)+0.  for tag in self.export_vars}
    
    def import_state(self, D):
        for tag, A in D.items():
            arr  = getattr(self, tag)
            arr *= 0.
            arr += A
        self.update_properties()
    
    def __stack_diags(self, tag):
        torch   =  self.args['torch']
        diags   =  [self.geom_vars[f'{tag}_{p}'][1:] for p in 'scn']
        matrix  =  torch.stack(diags).transpose(0,1) # (n,3) array
        assert matrix.shape == (self.n-1,3)
        return matrix

# ------------------------------------------------------------------------
#                  Quick Benchmark
# ------------------------------------------------------------------------

def quick_benchmark(CFD_solver_specialized, case_chosen = 'cRets', solve_energy = True):
    import  torch
    from    sys     import  path             as  sys_path
    from    os.path import  dirname, abspath

    torch.set_num_threads(1)
    print(f'torch.get_num_threads(): {torch.get_num_threads()}')

    sys_path.append(dirname(dirname(abspath(__file__))))
    from  a_dns.import_original_data  import  Import_dns_data
    sys_path.pop()

    imported_data  =  Import_dns_data.get()

    # print('all_cases = ', imported_data['all_cases'])

    for case in imported_data['all_cases']:
        if case!=case_chosen: continue
        data           =  imported_data['data'][case]
        args           =  { 'torch'                  :   torch                                                    ,
                            'y_grid'                 :   data['y']                                                ,
                            'Ret'                    :   data['Ret']                                              ,
                            'cp'                     :   data['cp']                                               ,
                            'A_sca_rho'              :   data['A_sca_r' ]                                         ,
                            'A_sca_mu'               :   data['A_sca_mu']                                         ,
                            'A_sca_k'                :   data['A_sca_k' ]                                         ,
                            'b_exp_rho'              :   data['b_exp_r' ]                                         ,
                            'b_exp_mu'               :   data['b_exp_mu']                                         ,
                            'b_exp_k'                :   data['b_exp_k' ]                                         ,
                            'tensor_device'          :  'cpu'                                                     ,
                            'as_tensor'              :   lambda x: torch.tensor(x,device='cpu',dtype=torch.double),
                            'ref_dns'                :   dict(u_dns = data['u_dns'], T_dns = data['T_dns'], mu_dns=data['mu_dns'], rho_dns = data['rho_dns']),
                            'bool_solve_energy'      :   solve_energy                                            ,
                            'compressible_correction':   False                                                   ,
                            'case'                   :  data['case']                                             ,
                            'constant_Sq_heat'       :  data['source_heat']                                      ,
                            'use_visc_heating'       :  data['visc_heat']                                        ,
                            'Pr_turb'                :  1.                                                       ,
                            'output_last'            :  True                                                     ,
                            'fprint'                 :  print                                                    ,
                          }
        print(case)
        m = CFD_solver_specialized(args)
        m.iterate()
        return m