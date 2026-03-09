# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    copy     import  deepcopy
import  sympy    as      sp
from    os.path  import  abspath, dirname
from    sys      import  path  as  sys_path

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(abspath(__file__)))
from  algebraic_optimizer  import  Optimized_Algebraic_Function
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

power  =  lambda a,b: a**b

def clean(D):
    if type(D) == dict:
        result = {}
        for k,v in D.items():
            if v:
                result[k] = clean(v)
    else:
        result = D
    return result

# ------------------------------------------------------------------------
#                  Sympy Derivations
# ------------------------------------------------------------------------

class Derive_Adj_Info_MK_Model:
    def __init__(self, hard_code = None, n_nested = 2, n_found = 2):
        #     k-eq:  0 = Pk -  rho e + ddy[(mu+mut/sigma_k) dkdy]                    + Source
        #     e-eq:  0 = Ce1 f1 e/k Pk - Ce2 r f2 e^2/k + ddy[(mu+mut/sigma_e)dedy]  + Source    
        
        self.n_nested, self.n_found = n_nested, n_found

        self.hard_code  =  hard_code
        self.__gen_vars()
        self.__mk_mueff()
        self.__mk_eqs  ()
        self.__mk_adjoint_matrices()
        self.get_adjoint_info  =  Optimized_Algebraic_Function({'D_terms' : deepcopy(self._adjoint_matrices),
                                                                'n_nested': self.n_nested                    ,
                                                                'n_found' : self.n_found                     ,
                                                                'known_vars': self.base_vars                 })

    def __gen_vars(self):
        
        all_tags  =  ['beta_k', 'beta_e', 'u_target',
                      'Ce1'   , 'Ce2'   , 'C_mu'    , 'sig_k'   , 'sig_e'   , 'mu_w'    , 'r_w'      , 'Ret'     ,
                      'u_imp', 'e_imp'  , 'k_imp'   , 'Ru_basis', 'Rk_basis', 'Re_basis', 'flag_e_BC'
                     ]
        
        for      key  in  ['u','k','e','mu','r','cd1','cd2', 'y']:
            for  p    in  'scn':
                all_tags.append(f'{key}_{p}')
        
        if self.hard_code:
            assert         all(map(lambda tag:     tag in all_tags      , self.hard_code)) # ensure all hard-coded variable are defined
            all_tags = list(filter(lambda tag: not tag in self.hard_code, all_tags))       # ensure all_tags are not in hard_code
        
        # vars contains the definition of every variable (symbol or float)
        self.vars      =  {tag: sp.symbols(tag, real=True) for tag in all_tags} 
        self.base_vars = list(all_tags)

        if self.hard_code:
            for tag in self.hard_code:
                assert type(self.hard_code[tag]) in [int, float]
                assert not (tag in self.vars     )
                assert not (tag in self.base_vars)
                self.vars[tag] = self.hard_code[tag]
        
        # add e_BC
        # e_BC = mu[0]/r[0]*k[1]/power(d[ 1], 2)
        #                                mu[0]  /          r[0] *           k[1]  / power(d[ 1], 2)
        # self.vars['flag_e_BC'] is added later
        self.vars['e_BC']  =  self.vars['mu_w']/self.vars['r_w']*self.vars['k_c'] / (self.vars['y_c']**2)
    
    def __mk_mueff(self):
        for p in 'scn':
            mu       =  self.vars[f'mu_{p}']
            mu_turb  =  self.__get_mu_turb(p)
            self.vars[f'mu_eff_u_{p}'] = mu + mu_turb
            self.vars[f'mu_eff_k_{p}'] = mu + mu_turb / self.vars['sig_k']
            self.vars[f'mu_eff_e_{p}'] = mu + mu_turb / self.vars['sig_e']

    def __calc_functions(self, p):
        
        r, mu, e, k, y  =  [self.vars[f'{key}_{p}'] for key in ['r', 'mu', 'e', 'k', 'y']]
        C_mu                =   self.vars['C_mu']
        r_w, mu_w, Ret      =   self.vars['r_w'], self.vars['mu_w'], self.vars['Ret']

        yplus               =  y*sp.sqrt(r/r_w)/(mu/mu_w)*Ret
        ReTurb              =  r*power(k, 2)/(mu*e)
        f_e2                =  (1.-2./9.*sp.exp(-power(ReTurb/6., 2)))*power(1-sp.exp(-yplus/5.), 2)
        ReTurb              =  sp.Max(ReTurb,1e-8)
        fmue                =  (1.-sp.exp(-yplus/70.))*(1.+3.45/power(ReTurb, 0.5))
        mut                 =   C_mu*fmue*r/e*power(k,2)

        return {'mut'  :  mut ,
                'f_e2' : f_e2 }
    
    def __get_mu_turb(self, p):
        return self.__calc_functions(p)['mut']
    
    def __get_f_e2(self, p):
        return self.__calc_functions(p)['f_e2']
    
    def __mk_eqs(self):
        def get_var(tag):
            result = self.vars[tag]
            if tag == 'e_s':
                result = result * (1-self.vars['flag_e_BC']) + self.vars['flag_e_BC'] * self.vars['e_BC']
            return result
        mk_deriv  =  lambda var  , i  : sum(map( lambda p: self.vars[f'cd{i}_{p}'] * get_var(f'{var}_{p}'),
                                                'scn'                                                     ))
        mk_diff   =  lambda mueff, var: (    mk_deriv(mueff,1)    *  mk_deriv(var,1) + 
                                         self.vars[f'{mueff}_c']  *  mk_deriv(var,2) )
        
        grad_u        =  mk_deriv('u',1)
        e             =  self.vars['e_c']
        k             =  self.vars['k_c']
        r             =  self.vars['r_c']
        Ce1           =  self.vars['Ce1']
        Ce2           =  self.vars['Ce2']
        beta_k        =  self.vars['beta_k']
        beta_e        =  self.vars['beta_e']
        
        f_e1          =  1.
        f_e2          =  self.__get_f_e2('c')
        mut           =  self.__get_mu_turb('c')

        Pk            =  mut*(grad_u**2)
        
        prod_k        =               Pk
        prod_e        =  Ce1*f_e1*e/k*Pk
        
        dest_k        =  -    r*      e
        dest_e        =  -Ce2*r*f_e2*(e**2)/k

        diff_k        =  mk_diff('mu_eff_k', 'k')
        diff_e        =  mk_diff('mu_eff_e', 'e')
        R_u           =  mk_diff('mu_eff_u', 'u') + 1

        R_k           =  prod_k + beta_k*dest_k + diff_k
        R_e           =  prod_e + beta_e*dest_e + diff_e

        # this should be calculated by the post-processing of the MK turbulence model
        # beta_k_expr   =  -(prod_k + diff_k)/dest_k
        # beta_e_expr   =  -(prod_e + diff_e)/dest_e
        # delta_k_expr  =  dest_k*(beta_k - 1)
        # delta_e_expr  =  dest_e*(beta_e - 1)

        u              =  self.vars['u_c']
        u_target       =  self.vars['u_target']
        u_imp          =  self.vars['u_imp']
        k_imp          =  self.vars['k_imp']
        e_imp          =  self.vars['e_imp']
        Ru_basis       =  self.vars['Ru_basis']
        Rk_basis       =  self.vars['Rk_basis']
        Re_basis       =  self.vars['Re_basis']

        res_u_contrib  =  (u-u_target)/Ru_basis
        res_k_contrib  =  (dest_k*(beta_k-1))/Rk_basis
        res_e_contrib  =  (dest_e*(beta_e-1))/Re_basis
        Jcost          =  u_imp*(res_u_contrib**2) + k_imp*(res_k_contrib**2) + e_imp*(res_e_contrib**2)

        self.eqs =  { 'R_u'         : R_u          ,
                      'R_k'         : R_k          ,
                      'R_e'         : R_e          ,
                      'Jcost'       : Jcost        }
    
    def __mk_adjoint_matrices(self):
        diff_w  =  lambda R:  clean({var                         : {p: R.diff(self.vars[f'{var}_{p}']) for p in 'scn'} for var in 'uke'})
        diff_b  =  lambda R:  clean({var.replace('beta_','Beta_'):     R.diff(self.vars[var])                          for var in ['beta_k','beta_e'] if var in self.base_vars})
        Jcost   =  self.eqs['Jcost']
        self._adjoint_matrices =  {'dRdw':  {f'R_{dof}': diff_w(self.eqs[f'R_{dof}']) for dof in 'uke'},
                                    'dRdB':  {f'R_{dof}': diff_b(self.eqs[f'R_{dof}']) for dof in 'uke'},
                                    'dJdw':  diff_w(Jcost)                                              ,
                                    'dJdB':  diff_b(Jcost)                                              }

if __name__ == '__main__':
    
    self = m = Derive_Adj_Info_MK_Model()
    import random
    random.seed(0)
    # D        = {k: (random.random()+0.5)*10  for k in m.base_vars}
    # m.get_adjoint_info(D)
