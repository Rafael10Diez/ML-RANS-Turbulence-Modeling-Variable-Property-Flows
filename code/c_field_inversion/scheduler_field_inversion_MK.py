# for i in {1..58}
# do
#     sleep 2
#     gnome-terminal --tab -- bash  -c "scheduler_field_inversion_MK.py; exec bash -i"
# done

# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path  import  isfile, isdir, abspath, basename, dirname
from    os.path  import  join      as  pjoin
from    os       import  mkdir     as  os_mkdir
from    sys      import  path      as  sys_path
from    copy     import  deepcopy
import  torch
torch.set_num_threads(1)

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(abspath(__file__)))
from  field_inversion_MK          import  Field_inversion_optimizer_MK_model
sys_path.pop()

sys_path.append(dirname(dirname(abspath(__file__))))
from  a_dns.import_original_data  import  Import_dns_data
from  misc.utils                  import  FPrint, reader
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

import_json = lambda fname: eval(' '.join(reader(fname)))

def mkdir_p(x):
    assert not isfile(x)
    if not isdir(x):
        mkdir_p(dirname(x))
        os_mkdir(x)
    assert isdir(x)

def priority_sort(A0, priorities, wildcard):
    A0    =  list(A0)
    A, B  =  deepcopy(A0), []
    for cards in priorities:
        all_f = []
        for i,a in enumerate(cards):
            if a == wildcard:
                assert not any(map(lambda row: row[i] == wildcard, A))
                all_f.append(lambda _: True)
            else:
                assert any(map(lambda row: row[i] == a, A))
                gen_f = lambda a,i: lambda row: row[i] == a
                all_f.append(gen_f(a,i))
        new, leave = [], []
        for row in A:
            if all(map(lambda f: f(row), all_f)):
                new  .append(row)
            else:
                leave.append(row)
        assert new
        B.extend(new)
        A = leave
    B.extend(A)
    assert sorted(A0, key=str) == sorted(B, key=str)
    return B

# ------------------------------------------------------------------------
#                  Arguments MK Solver
# ------------------------------------------------------------------------

class Build_arguments_fi_run:
    
    # define imported data as internal class-variable
    all_imported_data  =  Import_dns_data.get()
    ref_fi_results     =  None
    
    @staticmethod
    def get(case, hyperparams, all_b_mode):
        
        if Build_arguments_fi_run.ref_fi_results is None:
            Build_arguments_fi_run.ref_fi_results = import_json(pjoin(dirname(dirname(dirname(abspath(__file__)))), 'data', 'input', 'reference_field_inversion_results.pyjson'))
        
        assert all(map(lambda b_mode: b_mode in ['Beta_k', 'Beta_e'],
                       all_b_mode                                   ))
        
        if len(all_b_mode) == 1:
            subfolder = all_b_mode[0]
        else:
            get_hyparam  =  lambda b_mode: hyperparams[{'Beta_k': 'k_imp', 'Beta_e': 'e_imp'}[b_mode]]
            subfolder    =  '__'.join(map(lambda b_mode: f"{b_mode}__{get_hyparam(b_mode)}", all_b_mode))
        
        u_imp         =  hyperparams['u_imp']
        folder        =  pjoin(dirname(dirname(dirname(abspath(__file__))))                                                           ,
                               'data'                                                                                                 ,
                               'output'                                                                                               ,
                               'field_inversion'                                                                                      ,
                               subfolder                                                                                              ,
                               f'u_imp_{u_imp}'                                                                                       ,    
                               f"FI_run_{case}_u_imp_{hyperparams['u_imp']}_k_imp_{hyperparams['k_imp']}_e_imp_{hyperparams['e_imp']}")
        
        mkdir_p(dirname(folder)) # this will create folder up to u_imp_???
        
        fname_print  =  pjoin(folder, f"{basename(folder)}.log")
        tag_run = {(1  , 0  ):  'only_Beta_k'   ,
                   (0  , 1  ):  'only_Beta_e'   ,
                   (1  , 1  ):  'Ik_1_Ie_1'     ,
                   (1.5, 0.5):  'Ik_1.5_Ie_0.5' ,
                   (0.5, 1.5):  'Ik_0.5_Ie_1.5' }[hyperparams['k_imp'], hyperparams['e_imp']]
        
        if 'only_' in tag_run:
            assert  all_b_mode  ==  [tag_run.removeprefix('only_')]
        
        return     {'all_b_mode'            :  all_b_mode                                                           ,
                    'args_mk_solver'        :  Build_arguments_fi_run._mk_args_MK_solver(case)                      ,
                    'u_imp'                 :  hyperparams['u_imp']                                                 ,
                    'k_imp'                 :  hyperparams['k_imp']                                                 ,
                    'e_imp'                 :  hyperparams['e_imp']                                                 ,
                    'solve_sparse_adj'      :  True                                                                 ,
                    'all_ref_beta_distrib'  :  {b: Build_arguments_fi_run.ref_fi_results[tag_run][u_imp][case][f'beta_distrib_{b}' ] for b in 'KE'},
                    'all_ref_delta_distrib' :  {b: Build_arguments_fi_run.ref_fi_results[tag_run][u_imp][case][f'delta_distrib_{b}'] for b in 'KE'},
                    'fname_print'           :  fname_print                                                          ,
                    'folder'                :  folder                                                               }
    
    @staticmethod
    def _mk_args_MK_solver(case):
        data           =  Build_arguments_fi_run.all_imported_data['data'][case]

        ref_dns        =  dict(u_dns   = data['u_dns'  ],
                               T_dns   = data['T_dns'  ],
                               mu_dns  = data['mu_dns' ],
                               rho_dns = data['rho_dns'])
        
        return            { 'torch'                  :  torch                                                    ,
                            'y_grid'                 :  data['y']                                                ,
                            'Ret'                    :  data['Ret']                                              ,
                            'cp'                     :  data['cp']                                               ,
                            'A_sca_rho'              :  data['A_sca_r' ]                                         ,
                            'A_sca_mu'               :  data['A_sca_mu']                                         ,
                            'A_sca_k'                :  data['A_sca_k' ]                                         ,
                            'b_exp_rho'              :  data['b_exp_r' ]                                         ,
                            'b_exp_mu'               :  data['b_exp_mu']                                         ,
                            'b_exp_k'                :  data['b_exp_k' ]                                         ,
                            'tensor_device'          :  'cpu'                                                    ,
                            'as_tensor'              :  lambda x: torch.tensor(x,device='cpu',dtype=torch.double),
                            'ref_dns'                :  ref_dns                                                  ,
                            'case'                   :  data['case']                                             ,
                            'constant_Sq_heat'       :  data['source_heat']                                      ,
                            'use_visc_heating'       :  data['visc_heat']                                        ,
                            'Pr_turb'                :  1.                                                       ,
                            'fprint'                 :  lambda _: None                                           ,
                            'compressible_correction':  False                                                    , # used in different paper
                            'bool_solve_energy'      :  False                                                    , # not used during field inversion
                            'output_last'            :  False                                                    } # sparse solver is slower for [100,100] matrices

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    
    def mk_combinations():
        for         u_imp    in  map(lambda i: 10**i, range(-1,4)):
            for     case     in  Build_arguments_fi_run.all_imported_data['all_cases']:
                for all_imp  in [{'Beta_k': 1  , 'Beta_e': 1  },
                                 {'Beta_k': 1                 },
                                 {               'Beta_e': 1  },
                                 {'Beta_k': 0.5, 'Beta_e': 1.5},
                                 {'Beta_k': 1.5, 'Beta_e': 0.5}]:
                    yield u_imp, all_imp, case
    
    combinations  =  priority_sort(mk_combinations(), [[100      , '**any**'                 , 'cRets'   ],
                                                       [100      , {'Beta_k': 1             }, '**any**' ],
                                                       [100      , {             'Beta_e': 1}, '**any**' ],
                                                       ['**any**', {'Beta_k': 1, 'Beta_e': 1}, '**any**' ]], '**any**')
    
    for u_imp, all_imp, case  in  combinations:
        
        all_b_mode  =  list(all_imp.keys())

        k_imp       =  all_imp.get('Beta_k', 0)
        e_imp       =  all_imp.get('Beta_e', 0)

        print(f'Scheduler: (torch.get_num_threads(): {torch.get_num_threads()})')

        hyperparams  =  dict(u_imp = u_imp, k_imp = k_imp, e_imp = e_imp)

        if max(k_imp,e_imp)>1.1 and (case, u_imp) != ('cRets', 100):
            continue
        args                =  Build_arguments_fi_run.get(case, hyperparams, all_b_mode)
        args['iters_limit'] =  15_000

        if not isdir(args['folder']):
            os_mkdir(args['folder'])
            fprint  =  FPrint(args['fname_print'])
            m       =  Field_inversion_optimizer_MK_model(args, fprint)