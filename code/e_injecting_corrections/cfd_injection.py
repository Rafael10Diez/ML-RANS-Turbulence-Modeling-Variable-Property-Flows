# python /media/rafael/DATA/Dropbox/20220924_FIML_solver/code/e_injecting_corrections/cfd_injection.py  log_reg_L2_0.000e00__main_reg_L2_1.000e-04  0.9

# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path      import  abspath, isdir, dirname, basename
from    os.path      import  join                               as  pjoin
from    sys          import  path                               as  sys_path
from    os           import  mkdir                              as  os_mkdir
from    collections  import  OrderedDict, defaultdict                         # this is necessary when eval is called importating data 
from    sys          import  argv
from    copy         import  deepcopy
from    time         import  time
import  torch
torch.manual_seed(0)

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(dirname(abspath(__file__))))
from  misc .utils                                import  (deepdirname         ,
                                                          lmap                ,
                                                          lfilter             ,
                                                          reader_zipped_text  ,
                                                          listdir_full_files  ,
                                                          listdir_full_folders,
                                                          sorted_dict_by_key  ,
                                                          pop1                ,
                                                          zip_str_write       ,
                                                          FPrint              ,
                                                          Dictify_obj         )
from  c_field_inversion        .scheduler_field_inversion_MK  import  Build_arguments_fi_run
from  d_neural_network_mk_keq  .deep_learning                 import  Neural_Network
from  d_neural_network_mk_keq  .genetic_algorithm             import  Scan_folder
from  d_neural_network_mk_keq  .data_loaders                  import  Individual_data_loader
from  b_rans_solver.turb_models.mk_model                      import  CFD_Solver_MK_model
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

long_line           =  '-' * 130
long_pad            =  ' ' *  43

import_zipped_json  =  lambda fname: eval(' '.join(reader_zipped_text(fname)))

def lstrip_check(x, tag):
    assert (x[:len(tag)] == tag) and (x.count(tag)==1)
    return x.removeprefix(tag)

def get_avg(A):
    A = list(A)
    return sum(A)/len(A)

class Queue_vectors:
    def __init__(self, L):
        self.avg = 0.
        self.i   = 0
        self.L   = L
        self.A   = [0. for _ in range(self.L)]
    
    def update_getavg(self, x):
        A, i, L   =  self.A, self.i, self.L
        self.avg += (x - A[i]) / L
        A[i]      =  x + 0.  #  trigger copy
        self.i    =  (i+1)%L
        return  self.avg

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def get_cfd_solver_mk(case):
    args                       =  Build_arguments_fi_run._mk_args_MK_solver(case)
    args['bool_solve_energy']  =  True      #  enable solving the energy equation
    cfd_solver                 =  CFD_Solver_MK_model(args)
    cfd_solver.iterate()
    return cfd_solver

# ------------------------------------------------------------------------
#                  Weight relaxation factor methodology
# ------------------------------------------------------------------------

def relax_factor(args, tol_res = 1e-10, beta_clipper = float('inf')):
    
    delta, alpha         =  args['delta'], args['alpha']
    P2                   =  args['P']**2
    delta_P2             =  delta*P2
    alpha_delta_sq_sum   =  ((alpha*delta)**2).sum()

    delta_f          = lambda k:         delta_P2/(k+P2)
    R_eq             = lambda k:     (   delta_f(k)**2                ).sum()  -  alpha_delta_sq_sum  
    grad_R_eq        = lambda k:  -2*(  (delta_f(k)**2) / (k  + P2 )  ).sum()
    cost             = lambda k:    R_eq(k)**2
    grad             = lambda k:  2*R_eq(k)*grad_R_eq(k)
    k    =  k_old    =  1.
    res  =  res_old  =  cost(k)
    known_grad       =  None
    lr               =  1.
    iters            =  0

    while (res > tol_res) and (iters < 10000):
        iters += 1

        if known_grad is None:
            known_grad = grad(k)
        
        k        -=  lr*known_grad
        k         =  max(k, 1e-10)
        res       =  cost(k)

        if res < res_old:
            lr        *= 1.9
            res_old    = res
            k_old      = k + 0.
            known_grad = None
        else:
            k   = k_old + 0.
            lr *= 0.5
    
    beta_max = (delta*args['P']/(k+P2)).abs().max()

    if beta_max > beta_clipper:
        args            =  dict(args.items())
        args['alpha']  *=  0.9
        return relax_factor(args, tol_res)
    return delta_f(k), alpha, beta_max

# ------------------------------------------------------------------------
#                  Extended Scan_folder
# ------------------------------------------------------------------------
# extended scan_folder includes methods to load the neural network 
# stored in a folder, and also make predictions based on a dictionary
# with information about a cfd solver

class Scan_folder_predict(Scan_folder):
    def __init__(self, folder, device):
        super().__init__(folder, all_dset_benchmark = ['train'])
        assert self.finished_training
        self.device = device
        self.__already_loaded_network = False
        self.mk_info_reg()
    
    def load_network(self):
        assert not self.__already_loaded_network
        self.__already_loaded_network = True

        saved_net        =  sorted(filter(lambda x: x.endswith('_statedict_net.dat.zip'),
                                          listdir_full_files(pjoin(self.folder,'saved')))  ,
                                   key = lambda x: int(x.split('_epoch_')[1].split('_')[0]))[-1]
        
        self.initial_data     =  import_zipped_json(pjoin( self.folder          ,
                                                          'saved'               ,
                                                          'initial_data.dat.zip'))

        args_net                =  deepcopy(self.runner_args['args_net'])
        args_net['seed']        =  int(saved_net.split('seed_')[1].split('_')[0])
        args_net['n_features']  =  self.initial_data['loaders']['n_features']
        self.net                =  Neural_Network(args_net)
        self.net.load_state_dict(import_zipped_json(saved_net))
        self.net = self.net.to(self.device)
        self.net.eval()

        self.cases_dset         =  self.initial_data['args']['used_cases']
        self.order              =  list(pop1(list(set(tuple(loader['order'])  for loader in self.initial_data['loaders']['get_loader'].values()  ))))
        
        assert self.order == [ 'Y_star'     ,  'prod_k/Sk' ,
                               'u/Su'       ,  'dest_k/Sk' ,
                               'k/Mk'       ,  'diff_k/Sk' ,
                               'e/Me'       ,  'Ret_star'  ,
                               'r/r_w'      ,  'Su'        ,
                               'mu/mu_w'    ,  'Sk'        ,
                               'mu_t/mu_w'  ,  'Mk'        ]
        
    def predict_delta(self, m, Ru_basis, Rk_basis, Re_basis):
        dtype  =  torch.float
        
        # m is the cfd_solver object
        D           =  {'geom_vars'     :  {k:v.tolist() for k,v in m.geom_vars.items()} ,
                        'rans_u'        :  m.rans_u         .tolist() ,
                        'rans_MK_k'     :  m.rans_MK_k      .tolist() ,
                        'rans_MK_e'     :  m.rans_MK_e      .tolist() ,
                        'rans_rho_molec':  m.rans_rho_molec .tolist() ,
                        'rans_mu_molec' :  m.rans_mu_molec  .tolist() ,
                        'rans_mu_turb'  :  m.rans_mu_turb   .tolist() ,
                        'y'             :  m.y              .tolist() ,
                        'Ret'           :  m.Ret                      ,
                        'rans_MK_sig_k' :  m.rans_MK_sig_k            ,
                        'Ru_basis'      :  Ru_basis                   ,
                        'Rk_basis'      :  Rk_basis                   ,
                        'Re_basis'      :  Re_basis                   }

        args = dict(all_D   =  {m.args['case']: {'ini':D}}    ,
                    device  =  self.device                    ,
                    dtype   =  dtype                          ,
                    only_x  =  True                           ,
                    order   =  [ 'Y_star'     ,  'prod_k/Sk' ,
                                 'u/Su'       ,  'dest_k/Sk' ,
                                 'k/Mk'       ,  'diff_k/Sk' ,
                                 'e/Me'       ,  'Ret_star'  ,
                                 'r/r_w'      ,  'Su'        ,
                                 'mu/mu_w'    ,  'Sk'        ,
                                 'mu_t/mu_w'  ,  'Mk'        ])
        
        loader   =  Individual_data_loader(args)

        # delta had double normalization: delta /Rk_basis / max_training_set
        return self.net(loader.X_stack) * Rk_basis
    
    def mk_info_reg(self):
            log_reg      , main_reg        =  self.runner_args['argv_full'][-2:]
            assert 'p_loglayer'  in  log_reg
            assert 'p_change'    in  main_reg
            self.info_reg                  =  {'log_reg' : self.__parse_reg(log_reg ),
                                               'main_reg': self.__parse_reg(main_reg)}
            self.key_all_reg      =  ('log_reg' ,(self.info_reg['log_reg' ]['mode'],self.info_reg['log_reg' ]['value']),
                                      'main_reg',(self.info_reg['main_reg']['mode'],self.info_reg['main_reg']['value']))
            self.tag_all_reg  =  ('__'.join(f"{key}_{self.info_reg[key]['mode']}_{self.info_reg[key]['value']:.3e}" for key in ['log_reg', 'main_reg'])).replace('+','')
    
    @staticmethod
    def __parse_reg(text_reg):

        assert (text_reg.count('L1') + text_reg.count('L2')) <= 1

        if (not text_reg) or (text_reg[-4:] == 'None'):
            mode, value  =  'L2', 0
        else:
            mode, value  =  (  text_reg.split('x')[0].split('_')[-1]         ,
                               float(text_reg.split('x')[1].split('_')[ 0])  )

        assert mode in ['L1','L2']
        return {'mode' : mode ,
                'value': value}

# ------------------------------------------------------------------------
#                  Get runs for all kfolds (and regularization schemes)
# ------------------------------------------------------------------------

class Get_Production_Runs:
    def __init__(self, fprint, device, label_errors = ['train', 'test']):
        self.data = D = {}
        self.fprint   = fprint
        self.tags_to_keys_all_reg = {}

        for folder in listdir_full_folders(pjoin( deepdirname(abspath(__file__),3),
                                                 'data'                           ,
                                                 'output'                         ,
                                                 'nn_training'                    ,
                                                 'production_runs'                )):
            m  =  Scan_folder_predict(folder,device)
            if not m.kfold in D:
                D[m.kfold] = {}
            
            self.tags_to_keys_all_reg[m.tag_all_reg] = m.key_all_reg

            D[m.kfold][m.key_all_reg] = dict(m       =  m                                                          , 
                                             errors  =  {label: self.get_error(m, label) for label in label_errors})
        fprint(f'\n{long_line}\n{long_pad}Summary Regularization\n{long_line}')
        
        self.data = D = sorted_dict_by_key(D)
        for Kfold in D.keys():
            D[Kfold] = sorted_dict_by_key(D[Kfold])

            fprint(f'\n---------------------- Kfold {Kfold} ----------------------')
            for  key_reg, sub_d in  D[Kfold].items():
                m       =  sub_d['m'     ]
                errors  =  sub_d['errors']
                tags    =  ' '.join(f"({label} = {(err*100):9.3f}%)" for label, err in errors.items())
                fprint(f'    (tag: {m.tag_all_reg})    {tags}')
                
    @staticmethod
    def get_error(m, mode, metric = '|diff|/|total|:', n_avg = 10):
        A      =  m._A_log
        all_i  =  lfilter(lambda i: A[i][:25] == f'Prediction (dset = {mode:5s})', range(len(A)))
        assert len(all_i) >= n_avg
        return get_avg(float(A[i].split(metric)[1].split('%')[0])/100.  for i in  all_i[-n_avg:])

# ------------------------------------------------------------------------
#                  Score Features by relative weight inside neuron
# ------------------------------------------------------------------------

def score_mag_groups(m):
            wlog          =  m.net.A[0].weight
            order         =  m.order
            param_groups  =  m.initial_data['args']['args_net']['param_groups']

            assert wlog.shape == (param_groups, len(order)) == (3,14)


            wlog_norms  =  lmap(lambda i: float(torch.linalg.norm(wlog[i,:]).item()),
                                range(param_groups)                                 )

            assert all(map(lambda n: n<=4, wlog_norms))
            print('wlog_norms', wlog_norms)

            result    = {}
            for j,key in enumerate(order):
                # "i" is the index of the neuron
                # score relative importance inside each neuron
                rel          = [float(wlog[i,j].abs() /
                                      wlog[i,:].abs().sum()) for i in range(wlog.shape[0])]
                result[key]  =  get_avg(rel)
            
            assert abs(sum(result.values())-1) < 1e-6
            return result

class Score_features_net:
    def __init__(self, mode, n_trials = None, mag = None):
        self.mode      =  mode
        self.n_trials  =  n_trials
        self.mag       =  mag
    
    def __call__(self, m):
        as_tensor    =  lambda x, requires_grad=False: torch.tensor(x, dtype=torch.float, requires_grad=requires_grad)
        m            =  deepcopy(m)
        order        =  m.order
        use_dets     =  ['test','train']
        print(use_dets)
        f_cat        =  lambda tag,use_dets=use_dets: torch.cat([as_tensor(m.initial_data['loaders']['get_loader'][dset][tag]) for dset in use_dets])
        X_stack      =  f_cat('X_stack') +0.
        Y_stack      =  f_cat('Y_stack') +0.
        print('X_stack_max', torch.max(torch.max(X_stack.abs(),dim=0)[0],dim=0)[0])
        print('Y_stack_max', torch.max(torch.max(Y_stack.abs(),dim=0)[0],dim=0)[0])
        get_X_real   =  lambda       : X_stack + 0.
        get_loss     =  lambda X_real: torch.mean((Y_stack - m.net(X_real)).abs())

        data         =  {f'{var}_{dset}': f_cat(f'{var}_stack',[dset]).detach().numpy()+0.  for var in 'X'  for dset in ['train','test']}

        assert order ==  [ 'Y_star'     ,  'prod_k/Sk' ,
                           'u/Su'       ,  'dest_k/Sk' ,
                           'k/Mk'       ,  'diff_k/Sk' ,
                           'e/Me'       ,  'Ret_star'  ,
                           'r/r_w'      ,  'Su'        ,
                           'mu/mu_w'    ,  'Sk'        ,
                           'mu_t/mu_w'  ,  'Mk'        ]

        if self.mode == 'shuffle':

            result = {key: [] for key in order}
            
            for _ in range(self.n_trials):  
                for j, key in enumerate(order):
                    
                    X_real  =  get_X_real()

                    for ind_y in range(X_real.shape[1]):
                        slice   =  X_real[:, ind_y, j].view(-1)
                        slice  += -slice + slice[torch.randperm(slice.shape[0])] # remove existing value, and place new randperm
                    
                    result[key].append(float(get_loss(X_real).item()))

            result  =  {key: get_avg(val) for key,val in result.items()}
        
        elif self.mode == 'noise':
            
            get_mag  =  lambda x: torch.mean(x.abs(), dim=0, keepdim=True)
            result   = {key: [] for key in order}
            
            for _ in range(self.n_trials):  
                for j, key in enumerate(order):
                    
                    X_real  =  get_X_real()
                    X_mag   =  get_mag(X_real)

                    X_real[:,:,j] += 2*(torch.rand( X_real[:,:,j].shape,
                                           dtype  = X_real[:,:,j].dtype,
                                           device = X_real[:,:,j].device) - 0.5) * X_mag[:,:,j] * self.mag

                    result[key].append(float(get_loss(X_real).item()))

            result  =  {key: get_avg(val) for key,val in result.items()}
        
        elif self.mode == 'ig':
            ig      =  IntegratedGradients(m.net)
            result  =  {key: [] for key in order}

            for ind_y in range(data['X_train'].shape[1]):
                
                X_test     =  as_tensor((data['X_test'][:,ind_y:ind_y+1,:]+0.).tolist(), requires_grad=True)
                baselines = torch.mean(get_X_real(), dim = 0, keepdim = True)[:,ind_y:ind_y+1,:] + torch.zeros_like(X_test) # average of al variables, fill_out ind_y
                attr, _    =  ig.attribute(X_test, baselines = baselines, target=-1, n_steps = 1000, return_convergence_delta=True)
                importance =  torch.mean(attr.abs(), dim=0, keepdim = True)
                for j, key in enumerate(order):
                    result[key].extend(importance[:,:,j].view(-1).tolist())
            
            result  =  {key: get_avg(val) for key,val in result.items()}

        elif self.mode == 'gradients':

            X_real              =  as_tensor(get_X_real().tolist(), requires_grad=True)
            m.net.requires_grad =  False
    
            get_loss(X_real).backward()
    
            assert len(X_real.shape) == 3
    
            reduce_dims  = lambda x: torch.mean(x.abs(),dim=0)
            scores       =  reduce_dims(X_real)*reduce_dims(X_real.grad)
    
            assert scores.shape == (X_real.shape[1], len(order))
    
            result = {key: float(scores[:,j].sum().item()) for j,key in enumerate(order)}

        else:
            raise Exception(f'Unrecognized (mode: {self.mode})')
        
        return {key: val/sum(result.values()) for key,val in result.items()}

score_gradients  =  Score_features_net('gradients')
# score_shap       =  Score_features_net('shap')
score_shuffle    =  Score_features_net('shuffle', 100)
score_ig         =  Score_features_net('ig')

score_noise_5     =  Score_features_net('noise', 100, mag = 0.05)
score_noise_10    =  Score_features_net('noise', 100, mag = 0.1 )
score_noise_20    =  Score_features_net('noise', 100, mag = 0.2 )
score_noise_30    =  Score_features_net('noise', 100, mag = 0.3 )
score_noise_50    =  Score_features_net('noise', 100, mag = 0.5 )
score_noise_100   =  Score_features_net('noise', 100, mag = 1   )

# ------------------------------------------------------------------------
#                  Select runs with chosen regularization
# ------------------------------------------------------------------------

class Selected_Runs:
    def __init__(self, fprint, device, tag_all_reg):
        
        self.fprint        =  fprint

        # select data
        full_data          =  Get_Production_Runs(fprint, device)
        key_all_reg        =  full_data.tags_to_keys_all_reg[tag_all_reg]

        fprint(f'\n---------------------- Chosen reg_reg: {tag_all_reg} ----------------------')
        self.selected_data  =  {kfold: all_info[key_all_reg]  for kfold, all_info in  full_data.data.items()}
        for info in self.selected_data.values():
            info['m'].load_network()
        
        self.report_importance_features()

    def report_importance_features(self):
        fprint, selected_data  =  self.fprint, self.selected_data

        for tag_score, get_scores in [['mag_groups', score_mag_groups],
                                      ['ig'        , score_ig        ],
                                      ['gradients' , score_gradients ],
                                      ['shuffle'   , score_shuffle   ],
                                      ['noise_5'   , score_noise_5   ],
                                      ['noise_10'  , score_noise_10  ],
                                      ['noise_20'  , score_noise_20  ],
                                      ['noise_30'  , score_noise_30  ],
                                      ['noise_50'  , score_noise_50  ],
                                      ['noise_100' , score_noise_100 ],
                                     ]:
            
            fprint(f'\n{long_line}\n{long_pad}Summary relevant features (method = {tag_score})\n{long_line}')
            order = list(pop1(list(set(  tuple(info['m'].order)  for info   in selected_data.values()  ))))

            assert order ==  [ 'Y_star'     ,  'prod_k/Sk' ,
                               'u/Su'       ,  'dest_k/Sk' ,
                               'k/Mk'       ,  'diff_k/Sk' ,
                               'e/Me'       ,  'Ret_star'  ,
                               'r/r_w'      ,  'Su'        ,
                               'mu/mu_w'    ,  'Sk'        ,
                               'mu_t/mu_w'  ,  'Mk'        ]

            summary         =  {key: [] for key in order}
            all_local_score =  []
            for kfold, info in selected_data.items():

                fprint(f'\n---------------------- Kfold {kfold} (method = {tag_score}) ----------------------')
                local_score = dict(kfold   =  int(kfold)           ,
                                   method  =  tag_score            ,
                                   info    =  get_scores(info['m']))
                
                all_local_score.append(local_score)

                for key, score in local_score['info'].items():
                    fprint(f"    {key:20s}: (rel: {(score*100):6.3f} %)")
                    summary[key].append(score)
            
            fprint.only_to_file(f'### all_local_score__{tag_score} = {all_local_score}')
            
            fprint(f'\n\n---------------------- Overview (method = {tag_score}) ----------------------')

            text, f_rank  =  [], get_avg
            for j,key in enumerate(order):
                all_rel = summary[key]
                text.append([f_rank(all_rel), f"    {key:20s}: (avg: {(get_avg(all_rel)*100):6.3f} %) (min: {(min(all_rel)*100):6.3f} %) (max: {(max(all_rel)*100):6.3f} %) (j = {j:2d})"])
            text.sort(reverse = True)
            lmap(lambda p_line: fprint(p_line[1]), text)

# ------------------------------------------------------------------------
#                  Individual CFD Injector
# ------------------------------------------------------------------------

class Individual_cfd_injector:
    def __init__(self, m, alpha, kfold, dset, case, fprint, tol_u = 1e-5, n_start_energy = 50):
        self.alpha            =  alpha
        self.m                =  m
        super_copy            =  lambda x: deepcopy(x if type(x) == list  else (x+0.).tolist())
        # define inject & uncorrected models
        self.cfd_injected     =  get_cfd_solver_mk(case)
        self.cfd_uncorrected  =  get_cfd_solver_mk(case)
        Rk_basis              =  self.cfd_uncorrected.calc_budgets_MK()['Rk_basis']
        Re_basis              =  self.cfd_uncorrected.calc_budgets_MK()['Re_basis']
        u_dns                 =  self.cfd_uncorrected.as_tensor(self.cfd_uncorrected.args['ref_dns']['u_dns'])
        get_u_error           =  lambda cfd: float((cfd.rans_u - u_dns).abs().max()/u_dns.abs().max())
        self.u_error_baseline =  get_u_error(self.cfd_uncorrected)

        assert self.cfd_injected   .args['bool_solve_energy']
        assert self.cfd_uncorrected.args['bool_solve_energy']

        self.plot_args        =  {'alpha'             :  alpha                                                                                              , 
                                  'kfold'             :  kfold                                                                                              ,
                                  'dset'              :  dset                                                                                               ,
                                  'case'              :  case                                                                                               ,
                                  'u_baseline'        :  super_copy(self.cfd_uncorrected.rans_u                                                            ),
                                  'u_dns'             :  super_copy(u_dns                                                                                  ),
                                  'T_dns'             :  super_copy(self.cfd_uncorrected.args['ref_dns']['T_dns']                                          ),
                                  'delta_field_inv/Rk':  super_copy(m.initial_data['loaders']['get_loader'][dset]['ref_delta_fi'][case]['delta_k/Rk_basis']),
                                 }

        self.cfd_injected   .args['bool_solve_energy'] = False
        self.cfd_uncorrected.args['bool_solve_energy'] = False

        P_orig                =  self.cfd_uncorrected.rans_mu_turb * (self.cfd_uncorrected.get_grady(self.cfd_uncorrected.rans_u)**2)

        change_u              =  float('inf')
        iters                 =  0
        last_iter, t0, t1     =  iters, time(), time()

        delta_queue           =  Queue_vectors(n_start_energy//2)

        while (change_u > tol_u) or (iters<(2*n_start_energy)):
            iters += 1
            if iters > 500: break

            u_old = self.cfd_injected.rans_u + 0.

            # compute delta (uncorrected)
            #                                             Ru_basis
            delta = m.predict_delta(self.cfd_uncorrected, max(self.cfd_injected.rans_u), Rk_basis, Re_basis)

            assert delta.shape == (1,self.cfd_uncorrected.n,1)
            delta = self.cfd_injected.as_tensor(delta[0,:,0].tolist())

            delta_uncorrected  =  delta + 0. # trigger copy

            # relaxation factor methodology for delta
            delta, alpha_used, beta_max  =  relax_factor(dict(alpha = self.alpha, P = P_orig, delta = delta))

            # inject delta

            self.cfd_injected.delta_inject_K  =  delta_queue.update_getavg(delta)

            if iters>n_start_energy:
                self.cfd_injected.args['bool_solve_energy'] = True
            self.cfd_injected.iterate(assert_convergence=False)

            # inject rans_T
            self.cfd_uncorrected.rans_T = self.cfd_injected.rans_T + 0.
            self.cfd_uncorrected.update_properties()
            self.cfd_uncorrected.iterate()

            change_u = (self.cfd_injected.rans_u - u_old).abs().max()
            
            if  (time()-t1) > 10.:
                secs_iter = f"{((time()-t1)/(iters-last_iter)):.3f}" if (iters>last_iter) else None
                fprint(f'    Inject         (iters = {iters:6d}) (secs/iter = {secs_iter}) (change_u: {change_u:.3e}) (u_error: {(get_u_error(self.cfd_injected)*100):.3f} %) (u_error_baseline: {(self.u_error_baseline*100):.3f} %) (alpha_used: {alpha_used:.6f}) (beta_max: {beta_max:.6f})')
                last_iter, t1 = iters, time()
        self.cfd_injected.iterate()
        secs_iter = f"{((time()-t1)/(iters-last_iter)):.3f}" if (iters>last_iter) else None
        fprint(f'    Inject (final) (iters = {iters:6d}) (secs/iter = {secs_iter}) (change_u: {change_u:.3e}) (u_error: {(get_u_error(self.cfd_injected)*100):.3f} %) (u_error_baseline: {(self.u_error_baseline*100):.3f} %) (alpha_used: {alpha_used:.6f}) (beta_max: {beta_max:.6f}) (elapsed time: {(time()-t0):.3f})')
        self.u_error_improvement  =  self.u_error_baseline - get_u_error(self.cfd_injected) # positive is good

        self.plot_args['delta_uncorrected/Rk']  =  super_copy(delta_uncorrected                / Rk_basis)
        self.plot_args['delta_injected/Rk']     =  super_copy(self.cfd_injected.delta_inject_K / Rk_basis)
        self.plot_args['u_injected']            =  super_copy(self.cfd_injected.rans_u                   )
        self.plot_args['T_injected']            =  super_copy(self.cfd_injected.rans_T                   )


# ------------------------------------------------------------------------
#                  Process All CFD Injections
# ------------------------------------------------------------------------
class Get_all_cfd_injections:
    def __init__(self, tag_all_reg, alpha, device):

        self.output_folder   =  pjoin( deepdirname(abspath(__file__),3)               ,
                                      'data'                                          ,
                                      'output'                                        ,
                                      'cfd_injections'                                ,
                                      f'cfd_injections__alpha_{alpha}__{tag_all_reg}' ,
                                      )
        if not isdir(self.output_folder):
            os_mkdir(self.output_folder)
        self.fprint          =  FPrint(pjoin(self.output_folder,basename(self.output_folder)+'.log'))

        self.selected_data   =  Selected_Runs(self.fprint, device, tag_all_reg).selected_data

        self.all_cfd_injections  =  {}
        self.fprint(f'\n{long_line}\n{long_pad}Injection of CFD Corrections\n{long_line}')

        global_u_error_improvement = []

        for kfold, info  in  self.selected_data.items():
            all_u_error_improvement  =  []
            m                        =  info['m']
            assert  kfold  ==  m.kfold
            self.all_cfd_injections[kfold]  =  {}

            for     dset  in  ['valid', 'test']:
                self.all_cfd_injections[kfold][dset]  =  {}

                for case  in  m.initial_data['args']['used_cases'][dset]:
                    self.fprint(f"\n--------------------------- Iterating (kfold: {kfold}) (dset: {dset}) (case: {case}) ---------------------------\n")
                    self.all_cfd_injections[kfold][dset][case]  = inj =   Individual_cfd_injector(m, alpha, kfold, dset, case, self.fprint)
                    all_u_error_improvement.append(inj.u_error_improvement)
            
            all_u_error_improvement  =  get_avg(all_u_error_improvement)
            global_u_error_improvement.append(all_u_error_improvement)

            self.fprint(f'\n--------------------------- all_u_error_improvement: {(all_u_error_improvement*100):.3f} %  ---------------------------\n')
        
        global_u_error_improvement  =  get_avg(global_u_error_improvement)
        self.fprint(f'\n--------------------------- global_u_error_improvement: {(global_u_error_improvement*100):.3f} %  ---------------------------\n')
        
        zip_str_write(pjoin(self.output_folder, 'cfd_injections.dat'),
                      Dictify_obj.get(self)                          ,
                      check_path = False                             )

# ------------------------------------------------------------------------
#                  Main Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    device               =  'cpu' # cuda is unnecessary to make predictions
    tag_all_reg, alpha   =  argv[1:]
    alpha                =  float(alpha)
    all_cfd_injections   =  Get_all_cfd_injections(tag_all_reg, alpha, device)
    




