# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  torch
from    copy   import  deepcopy

# ------------------------------------------------------------------------
#                  All Data Loaders
# ------------------------------------------------------------------------

class All_Data_Loaders:
    def __init__(self, args):
        
        order  =  args['order']
        
        assert order == [ 'Y_star'     ,  'prod_k/Sk' ,
                          'u/Su'       ,  'dest_k/Sk' ,
                          'k/Mk'       ,  'diff_k/Sk' ,
                          'e/Me'       ,  'Ret_star'  ,
                          'r/r_w'      ,  'Su'        ,
                          'mu/mu_w'    ,  'Sk'        ,
                          'mu_t/mu_w'  ,  'Mk'        ]
        
        self.n_features  =  len(order)
        self.get_loader  =  {}
        for tag in ['train', 'valid', 'test']:
            if args['get_all_D'][tag]:
                self.get_loader[tag]  =  Individual_data_loader({'device'   :  args['device']         ,
                                                                 'dtype'    :  args['dtype']          ,
                                                                 'order'    :  order                  ,
                                                                 'all_D'    :  args['get_all_D'][tag] ,
                                                                 'only_x'   :  False                  })

# ------------------------------------------------------------------------
#                  Individual Data Loaders
# ------------------------------------------------------------------------

class Individual_data_loader:
    def __init__(self, args):
        # args.keys -> device, dtype, order, all_D, only_x
        
        # all_D contains all dictionaries with the input data for each ML run
        #    data must be gathered by an external function
        
        self.args                  =  args
        all_D                      =  args['all_D']
        self.order  =  order       =  args['order']
        ny                         =  len(list(all_D.values())[0]['ini']['y'])

        # assert  'cuda'       in  args['device']
        assert  torch.float  ==  args['dtype']
        assert  100          ==  ny
        assert  all(map(lambda d: len(d['ini']['y']) == ny,
                        all_D.values()                    ))
        
        assert order == [ 'Y_star'     ,  'prod_k/Sk' ,
                          'u/Su'       ,  'dest_k/Sk' ,
                          'k/Mk'       ,  'diff_k/Sk' ,
                          'e/Me'       ,  'Ret_star'  ,
                          'r/r_w'      ,  'Su'        ,
                          'mu/mu_w'    ,  'Sk'        ,
                          'mu_t/mu_w'  ,  'Mk'        ]
        
        as_tensor     =  lambda x        :  torch.tensor(x                       ,
                                                         device = args['device'] ,
                                                         dtype  = args['dtype']  )
        
        mk_zeros      =  lambda last=None:  torch.zeros((len(all_D), ny, last)   ,
                                                         device = args['device'] ,
                                                         dtype  = args['dtype']  )

        self.X_stack  =  mk_zeros(last = len(order))
        
        if not args['only_x']:
            self.Y_stack       =  mk_zeros(last = 1)
            self.ref_delta_fi  =  {}
        
        for     i,case      in  enumerate(all_D.keys()):
            D  =  all_D[case]
            if not args['only_x']:
                self.Y_stack[i,:,0]   =  delta               =  self.__get_Y(D['end'], as_tensor)
                self.ref_delta_fi[case]                      =  deepcopy(D['end'])
                self.ref_delta_fi[case]['delta_k/Rk_basis']  =  (delta + 0.).tolist()
            
            terms  =  self.__get_terms(D['ini'], as_tensor) # order of terms is not used
            for k,oo  in  enumerate(self.order):
                self.X_stack[i,:,k] = terms[oo]

    @staticmethod
    def __get_Y(D, as_tensor):
        r              =  as_tensor(D['rans_rho_molec'])
        e              =  as_tensor(D['rans_MK_e']     )
        beta_k         =  as_tensor(D['beta_distrib_K'])
        dest_k         =  -r*e
        res_k_contrib  =  (dest_k*(beta_k-1))/D['Rk_basis'] # delta / Rk_basis
        return res_k_contrib

    @staticmethod
    def __get_terms(D, as_tensor):
        
        grad      =  lambda val: sum(as_tensor(D['geom_vars'][f'cd1_{p}'])*val[D['geom_vars'][f'i_{p}']] for p in 'scn')
        nabla     =  lambda val: sum(as_tensor(D['geom_vars'][f'cd2_{p}'])*val[D['geom_vars'][f'i_{p}']] for p in 'scn')

        u         =  as_tensor(D['rans_u'])
        k         =  as_tensor(D['rans_MK_k'])
        e         =  as_tensor(D['rans_MK_e'])

        r         =  as_tensor(D['rans_rho_molec'])
        r_w       =  r[0]

        mu        =  as_tensor(D['rans_mu_molec'])
        mu_w      =  mu[0]

        mu_turb   =  as_tensor(D['rans_mu_turb'])
        
        Ret_star  =  torch.sqrt(r/r_w)/(mu/mu_w)*D['Ret']

        mueff_k   =  mu + mu_turb / D['rans_MK_sig_k']

        prod_k    =  mu_turb * (grad(u)**2)
        dest_k    =  -r*e
        diff_k    =  grad(mueff_k) * grad(k) + mueff_k * nabla(k)

        Ru_basis  =  D['Ru_basis']
        Rk_basis  =  D['Rk_basis']

        Me        =  Rk_basis   / r_w
        Mk        =  r_w*(Me**2)/ D['Re_basis']
        y         =  as_tensor(D['y'])

        return  {'Y_star'   :  y*Ret_star   ,        'prod_k/Sk': prod_k / Rk_basis , 
                 'u/Su'     :  u/Ru_basis   ,        'dest_k/Sk': dest_k / Rk_basis ,
                 'k/Mk'     :  k/Mk         ,        'diff_k/Sk': diff_k / Rk_basis , 
                 'e/Me'     :  e/Me         ,        'Ret_star' : Ret_star          , 
                 'r/r_w'    :  r      /r_w  ,        'Su'       : Ru_basis          , 
                 'mu/mu_w'  :  mu     /mu_w ,        'Sk'       : Rk_basis          , 
                 'mu_t/mu_w':  mu_turb/mu_w ,        'Mk'       : Mk                }  