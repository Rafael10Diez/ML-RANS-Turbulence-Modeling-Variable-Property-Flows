# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path              import  isfile, isdir, abspath, dirname, basename
from    os.path              import  join  as  pjoin 
from    sys                  import  path  as  sys_path
from    scipy.sparse         import  csc_matrix
from    scipy.sparse.linalg  import  spsolve
from    numpy                import  prod  as  np_prod
from    time                 import  time
from    collections          import  OrderedDict

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(abspath(__file__)))
from  expressions_discrete_adjoint_method_MK  import  Derive_Adj_Info_MK_Model
sys_path.pop()

sys_path.append(dirname(dirname(abspath(__file__))))
from  b_rans_solver.turb_models.mk_model      import  CFD_Solver_MK_model
from  misc.utils import (lmap            ,
                         improved_pformat,
                         zip_str_write   ,
                         Copy_Zipped     ,
                         Dictify_obj     )
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

power    =  lambda a  , b:  a**b
get_avg  =  lambda x     :  sum(x)/len(x)

def apply_f(x, f):
    if type(x) == dict:
        x, items  =  {}, list(x.items())
        for k,v in items:
            x[k] = apply_f(v, f)
    else:
        assert hasattr(x,'shape') or (type(x) == list)
        x = f(x) # f is usually as_tensor, or a function that works with lists
    return x

# ------------------------------------------------------------------------
#                  Field Inversion Optimizer MK Model
# ------------------------------------------------------------------------

class Field_inversion_optimizer_MK_model:
    def __init__(self, args, fprint):
        
        self._last_saved  =  float('-inf')
        self.n_accepted   = 0
        self.n_rejected   = 0

        self.args         =  args
        self.fprint       =  fprint
        self.print_header()
        
        assert not ('b_mode'             in  args)
        assert not ('ref_beta_distrib'   in  args)
        assert not ('ref_delta_distrib'  in  args)

        self.all_b_mode  =  all_b_mode  =  args['all_b_mode']
        self.solve_sparse_adj           =  args['solve_sparse_adj']
        
        # a) setup cfd solver
        # a.1) load betas
        self.all_tag_beta_distrib  =  lmap(lambda b_mode: {'Beta_k':  'beta_distrib_K',
                                                           'Beta_e':  'beta_distrib_E'}[b_mode], all_b_mode)
        
        # a.2) initialize & iterate
        self.cfd_solver  =  CFD_Solver_MK_model(args['args_mk_solver'])
        self.cfd_solver.iterate()

        # b) setup discrete adjoint method
        n                          =  self.cfd_solver.n
        torch                      =  self.cfd_solver.args['torch']
        self.all_ref_beta_distrib  =  apply_f(self.args['all_ref_beta_distrib'] , self.cfd_solver.as_tensor)
        self.all_ref_delta_distrib =  apply_f(self.args['all_ref_delta_distrib'], self.cfd_solver.as_tensor)
        # print(self.all_ref_beta_distrib)
        # b.1) variables for the discrete adjoint method
        #     cd1 and cd2 (gradient and concavity coefficients respectively)
        self.adj_variables              =  {tag: self.cfd_solver.geom_vars[tag][1:] for tag in self.cfd_solver.geom_vars.keys() if tag[:2]=='cd'}

        #     u_dns
        self.adj_variables['u_target']  =  self.cfd_solver.as_tensor(self.cfd_solver.args['ref_dns']['u_dns'])[1:]

        assert len(self.adj_variables) == 7

        #     indexes north, center, south
        self.adj_inds  =  {p: self.cfd_solver.geom_vars[f'i_{p}'][1:] for p in 'scn'}
        
        def quick_check_adj_inds():
            adj_mask_pos                     =   {p: (self.adj_inds[p]>0) for p in 'scn'}
            assert  adj_mask_pos['c'].sum() ==  adj_mask_pos['c'].numel()
            assert  adj_mask_pos['n'].sum() ==  adj_mask_pos['n'].numel()
            assert (adj_mask_pos['s'].sum() == (adj_mask_pos['s'].numel()-1)) and not float(adj_mask_pos['s'][0].item())
        quick_check_adj_inds()

        #     y-coordinates
        for p in 'scn':
            self.adj_variables[f'y_{p}'] = self.cfd_solver.y[self.adj_inds[p]]
        
        # flag_e_BC
        self.adj_variables['flag_e_BC']    = torch.zeros_like(self.adj_variables['u_target'])
        self.adj_variables['flag_e_BC'][0] = 1. # [0] is the position of the first cell, since len(self.adj_variables['flag_e_BC']) == (n-1)

        #     functions for evaluation
        #         (sympy leaves a placeholder for the functions to utilize)
        one_                             =  self.cfd_solver.as_tensor(1.)
        half_                            =  0.5*one_
        self.adj_variables['Abs' ]       =  torch.abs
        self.adj_variables['sign']       =  torch.sign
        self.adj_variables['exp' ]       =  torch.exp
        self.adj_variables['Heaviside']  =  lambda a  : torch.heaviside(a, half_)
        self.adj_variables['Max']        =  lambda a,b: torch.maximum(a*one_, b*one_)
        self.adj_variables['sqrt']       =  torch.sqrt

        # b.2) hard-code constant variables for the field inversion optimizer
        #     (any reference variable: Ret, MK model constants, u_importance, Ru_basis, etc.)

        set(self.all_b_mode).issubset({'Beta_k','Beta_e'}) # planets.issubset(galaxy)
        
        self.hard_code  =  lambda: {b_mode.lower(): 1 for b_mode in ({'Beta_k','Beta_e'}-set(self.all_b_mode))} # pick opposite
        self.hard_code  =  self.hard_code()
        # self.hard_code['flag_e_BC'] = 0

        #     const function hyper-parameters
        for tag in ['u_imp', 'k_imp', 'e_imp']:
            self.hard_code[tag]     =  args[tag]
        
        #     scaling basis (Ru_basis, Rk_basis, etc.)
        self.hard_code['Ru_basis']  =  float(self.adj_variables['u_target'].abs().max())

        budgets_  =  self.cfd_solver.calc_budgets_MK()
        for tag in ['Rk_basis','Re_basis']:
            self.hard_code[tag]     =  budgets_[tag]
        
        #     Mk turbulence model constants
        for tag in ['Ce1' , 'Ce2', 'C_mu', 'sig_k', 'sig_e']:
             self.hard_code[tag] = getattr(self.cfd_solver, f'rans_MK_{tag}')
        
        #     flow parameters
        self.hard_code['Ret' ]     =  self.cfd_solver.Ret
        self.hard_code['mu_w']     =  float(self.cfd_solver.rans_mu_molec [0])
        self.hard_code['r_w' ]     =  float(self.cfd_solver.rans_rho_molec[0])

        #     check that [k_imp, e_imp] were properly implemented
        assert  [self.hard_code['k_imp'], self.hard_code['e_imp']]  in  {('Beta_k'         ,): [[1  , 0  ]],
                                                                         ('Beta_e'         ,): [[0  , 1  ]],
                                                                         ('Beta_k','Beta_e',): [[1.5, 0.5],
                                                                                                [0.5, 1.5],
                                                                                                [1  , 1  ]],
                                                                        }[tuple(self.all_b_mode)]
        assert self.hard_code['u_imp'] in [0.1, 1, 10, 100, 1000]
        
        # b.3) build engine to obtain discrete adjoint matrices
        self.engine_adj_info  =  Derive_Adj_Info_MK_Model(self.hard_code)
        
        # b.4) allocate matrices for the discrete adjoint method (dRdW_t and dJdB are built at runtime)
        mk_zeros   =  lambda shape: self.cfd_solver.as_tensor([0. for _ in range(np_prod(shape))]).view(shape)
        
        n_betas    =  len(self.all_b_mode)
        self.dRdB  =  mk_zeros((3*(n-1), n_betas*(n-1)))
        self.dJdw  =  mk_zeros( 3*(n-1)                )
        self.dJdB  =  mk_zeros(          n_betas*(n-1) )
        self.all_lims_tag_beta = [[i_offset*(n-1), (i_offset+1)*(n-1)]  for i_offset in  range(n_betas)]
        assert self.all_lims_tag_beta[-1][-1] == (n_betas*(n-1))

        self.optimize()
    
    def print_header(self):
        comment  =  '# '
        line     =  comment + '-'*72
        pad      =  ' '*17
        self.fprint(line)
        self.fprint(f'{comment}{pad}Input Arguments')
        self.fprint(line)
        self.fprint(f"{comment}Begin_Input_Arguments")
        self.fprint(improved_pformat(self.args))
        self.fprint(f"{comment}End_Input_Arguments")
        self.fprint('')
        self.fprint(line)
        self.fprint(f'{comment}{pad}Iterations')
        self.fprint(line)
        self.fprint('')

        Copy_Zipped( dirname(dirname(abspath(__file__))),  # code_folder
                     self.args['folder']                )
    
    def optimize(self):
        self.t0 = self.t1 = time()
        self.lr           =  1e-3
        self.lr_tol       =  1e-10
        self.k_more       =  1.2
        self.k_less       =  0.5
        self.c1_mom_ref   =  0.9 if self.hard_code['u_imp'] > 2 else 0.5
        self.c1_mom       =  0.    # this will automatically initialize grad_memo

        self.old_state      =  self.cfd_solver.export_state()
        self.J_old          =  self.get_Jcost()
        self.known_grad     =  None
        self.freq_print     =    100
        self.freq_save      =    500 # self.freq_print * (10**30)
        self.iters = self.last_iters = 0

        assert self.freq_save % self.freq_print == 0

        iters_limit = self.args['iters_limit']  if 'iters_limit' in self.args else  float('inf')
        print(f"Field inversion run:  (iters_limit = {iters_limit})")
        
        while self.lr > self.lr_tol and (self.iters < iters_limit):
            
            if self.known_grad is None:
                self.known_grad  =  self.get_grad_betas() + 0.
            
            if not self.iters % self.freq_print:
                self.print_progress()
            
            if not self.iters % self.freq_save:
                self.save_progress()
            
            self.iters += 1

            if self.c1_mom:
                self.grad_memo  *=     self.c1_mom
                self.grad_memo  +=  (1-self.c1_mom)*self.known_grad
            else:
                self.grad_memo   =  self.known_grad + 0.
            
            for tag_beta, lims_tag_beta in zip(self.all_tag_beta_distrib, 
                                               self.all_lims_tag_beta   ):
                ia,Lb             =  lims_tag_beta
                beta_distrib      =  getattr(self.cfd_solver, tag_beta)
                beta_distrib[1:] -=  self.lr * self.grad_memo[ia:Lb]
            
            self.cfd_solver.iterate()
            
            J_new                =  self.get_Jcost()

            if J_new < self.J_old:
                self.lr         *=  self.k_more
                self.c1_mom      =  self.c1_mom_ref
                self.old_state   =  self.cfd_solver.export_state()
                self.J_old       =  J_new
                # new betas are accepted, we do not know the new gradient
                self.known_grad  =  None
                self.n_accepted += 1
            else:
                self.lr     *=  self.k_less
                self.c1_mom  =  0
                self.cfd_solver.import_state(self.old_state)
                # self.cfd_solver.iterate()
                # self.known_grad is kept implicitly
                self.n_rejected += 1
        
        self.finished_running = True
        self.print_progress()
        self.save_progress()
    
    def save_progress(self):
        
        if self.iters > self._last_saved:
            
            fname = pjoin(self.args['folder'],basename(self.fprint.fname).removesuffix('.log')+f"_state_iter_{self.iters}.dat")
            zip_str_write(fname, Dictify_obj.get(self))
            self.fprint(f'-> Saved file: {fname}')
            self._last_saved = self.iters + 0

    def print_progress(self):
        
        beta_change   =  []
        delta_change  =  []
        delta_R_peak  =  []

        for  b_mode, tag_beta in  zip(self.all_b_mode          ,
                                      self.all_tag_beta_distrib):
            
            beta_distrib  =  getattr(self.cfd_solver, tag_beta)
            # let ref_scale = 1 
            beta_change.append(float((beta_distrib[1:] - self.all_ref_beta_distrib[b_mode[-1].upper()][1:]).abs().max())) # /self.ref_beta_distrib.abs().max()
        
            if b_mode == 'Beta_k':
                dest   = -self.cfd_solver.rans_rho_molec * self.cfd_solver.rans_MK_e
                basis  =  self.hard_code['Rk_basis']

            elif b_mode == 'Beta_e':

                dest   =  self.cfd_solver.calc_budgets_MK(return_Dest_E = True)
                basis  =  self.hard_code['Re_basis']

            else:
                raise Exception(f"Unrecognized option (b_mode: {b_mode})")

            delta        =  dest*(beta_distrib-1)

            delta_change.append(float((delta[1:] - self.all_ref_delta_distrib[b_mode[-1].upper()][1:]).abs().max()/basis))
            delta_R_peak.append(float((delta[1:]                                                     ).abs().max()/basis))

        u_error      =  float((self.cfd_solver.rans_u[1:]-self.adj_variables['u_target']).abs().max()/self.hard_code['Ru_basis'])
        if self.iters != self.last_iters:
            secs_iter  =  f"{((time()-self.t1)/max(1e-10,self.iters-self.last_iters)):6.3f}"
            ratio_accepted  =  self.n_accepted / (self.n_accepted + self.n_rejected)
        else:
            secs_iter  =  '  None'
            ratio_accepted  = 0
        
        self.fprint(f'iter: {self.iters:6d} (Jcost: {self.J_old:.6e}) (lr = {self.lr:.3e}) (elapsed time: {(time()-self.t0):6.3f}) (secs/iter: {secs_iter}) (n_accepted: {(ratio_accepted*100):.0f}%) (u_error: {(u_error*100):9.6f} %) (beta_change: {(get_avg(beta_change)*100):9.6f} %) (delta_change: {(get_avg(delta_change)*100):9.6f} %) (delta_peak/R: {(get_avg(delta_R_peak)*100):9.6f} %)')
        self.t1         = time()
        self.last_iters = self.iters
        
        self.n_accepted = 0
        self.n_rejected = 0
        
    def get_Jcost(self):
        # code symbolic solver:
        #     res_u_contrib  =  (u-u_target)/Ru_basis
        #     res_k_contrib  =  (dest_k*(beta_k-1))/Rk_basis
        #     res_e_contrib  =  (dest_e*(beta_e-1))/Re_basis
        #     Jcost          =  u_imp*(res_u_contrib**2) + k_imp*(res_k_contrib**2) + e_imp*(res_e_contrib**2)

        u_imp          =  self.args['u_imp']
        k_imp          =  self.args['k_imp']
        e_imp          =  self.args['e_imp']

        u              =  self.cfd_solver.rans_u[1:]
        u_target       =  self.adj_variables['u_target']

        res_u_contrib  =  (u-u_target)/self.hard_code['Ru_basis']

        Jcost          =  float((u_imp*(res_u_contrib**2)).sum())
        r              =  self.cfd_solver.rans_rho_molec
        e              =  self.cfd_solver.rans_MK_e

        if k_imp:
            dest_k          = -r*e
            beta_k          =  self.cfd_solver.beta_distrib_K

            res_k_contrib   =  (dest_k*(beta_k-1))/self.hard_code['Rk_basis']

            Jcost          += float((k_imp*(res_k_contrib[1:]**2)).sum())
        
        if e_imp:
            dest_e         =  self.cfd_solver.calc_budgets_MK(return_Dest_E = True)
            beta_e         =  self.cfd_solver.beta_distrib_E

            res_e_contrib  =  (dest_e*(beta_e-1))/self.hard_code['Re_basis']

            Jcost         +=  float((e_imp*(res_e_contrib[1:]**2)).sum())

        return Jcost
    
    def get_grad_betas(self):
        self.__fill_matrices_adj()
        if self.solve_sparse_adj:
            x = spsolve(  self.dRdW_t , # A
                        -self.dJdw   )
            x = self.cfd_solver.as_tensor(x)
        else:
            x  =  self.cfd_solver.args['torch'].linalg.solve( self.dRdW_t, 
                                                             -self.dJdw  )
        return (x.view(1,-1) @ self.dRdB).view(-1) + self.dJdB
    
    def __fill_matrices_adj(self):
        # discrete adjoint method
        n           =  self.cfd_solver.n

        for var, arr in [['u' , self.cfd_solver.rans_u        ],
                         ['k' , self.cfd_solver.rans_MK_k     ],
                         ['e' , self.cfd_solver.rans_MK_e     ],
                         ['mu', self.cfd_solver.rans_mu_molec ],
                         ['r' , self.cfd_solver.rans_rho_molec]]:
            for p in 'scn':
                self.adj_variables[f'{var}_{p}'] = arr[self.adj_inds[p]]
        
        for b_mode, tag_beta in zip(self.all_b_mode, self.all_tag_beta_distrib):
            self.adj_variables[b_mode.lower()] = getattr(self.cfd_solver, tag_beta)[1:]
        
        adj_info  =  self.engine_adj_info.get_adjoint_info(self.adj_variables)
        
        dRdw        =  {'data': [], 'row_inds': [], 'col_inds': []}
        self.dRdB  *=  0.
        self.dJdw  *=  0.
        self.dJdB  *=  0.

        # offset = base_offset - positions_condensed
        offset  = {'R_u': 0-1, 'R_k': n-2, 'R_e': 2*n-3,
                     'u': 0-1,   'k': n-2,   'e': 2*n-3}
        for         R_var  in  ['R_u', 'R_k', 'R_e']:
            row_inds  =  list(range(1 + offset[R_var] ,
                                    n + offset[R_var]))
            for     v_var  in  [  'u',   'k',   'e']:
                for p      in  'scn':
                        data      =  adj_info['dRdw'][R_var][v_var][p] 
                        col_inds  =  self.adj_inds[p] + offset[v_var]
                        if p == 's':
                            dRdw['data'    ].extend(data    .tolist()[1:])
                            dRdw['row_inds'].extend(row_inds         [1:])
                            dRdw['col_inds'].extend(col_inds.tolist()[1:])
                        else:
                            dRdw['data'    ].extend(data    .tolist())
                            dRdw['row_inds'].extend(row_inds         )
                            dRdw['col_inds'].extend(col_inds.tolist())
                        # dRdw['data'    ].extend(data    [adj_mask_pos[p]].tolist())
                        # dRdw['row_inds'].extend(row_inds[adj_mask_pos[p]].tolist())
                        # dRdw['col_inds'].extend(col_inds[adj_mask_pos[p]].tolist())

            for b_key, lims_tag_beta in zip(self.all_b_mode, self.all_lims_tag_beta):
                if b_key in adj_info['dRdB'][R_var]:
                    ia,Lb = lims_tag_beta
                    self.dRdB[1+offset[R_var]:n+offset[R_var], ia:Lb] += adj_info['dRdB'][R_var][b_key].diag()
        
        #    [col_inds, row_inds] are switched, because we need dRdW_t
        self.dRdW_t = csc_matrix((dRdw['data'], (dRdw['col_inds'], dRdw['row_inds'])), shape=(3*n-3, 3*n-3))
        if not self.solve_sparse_adj:
            self.dRdW_t = self.cfd_solver.as_tensor(self.dRdW_t.A)
        
        for     v_var  in  [  'u',   'k',   'e']:
            for p in adj_info['dJdw'][v_var].keys():
                assert p == 'c'
                self.dJdw[offset[v_var]+1:offset[v_var]+n]  +=  adj_info['dJdw'][v_var][p]
        
        for b_key, lims_tag_beta in zip(self.all_b_mode, self.all_lims_tag_beta):
            ia,Lb             =  lims_tag_beta
            self.dJdB[ia:Lb] +=  adj_info['dJdB'][b_key]
