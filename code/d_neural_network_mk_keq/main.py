
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path   import  abspath, isdir, dirname
from    os.path   import  join                     as  pjoin
from    sys       import  path                     as  sys_path
from    sys       import  argv
from    copy      import  deepcopy

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(         dirname(abspath(__file__)))
from ml_runner   import  ML_Runner
sys_path.pop()

sys_path.append(dirname(dirname(abspath(__file__))))
from  misc .utils                 import  deepdirname, lfilter
from  a_dns.import_original_data  import  Import_dns_data
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------


def lstrip_sure(A, b):
    assert len(A) > len(b)
    assert A[:len(b)] == b
    return A[len(b):]

def lsum(A):
    result = []
    for x in A:
        assert type(x) == list
        result += x
    return result

def parse_layers(layers):
    # layers have form: T2.S2__T5__P__T4.R
    layers  =  layers.split('__')
    assert str(layers).count('_') == 0
    for i,c in enumerate(layers):
        layers[i] = lsum(map(lambda c: [c[0]]*int(c[1:] or 1),
                             c.split('.')                    ))
    return layers

# ------------------------------------------------------------------------
#                  Build K-fold Trials
# ------------------------------------------------------------------------

class Build_Kfold:
    __banned       =  ['t31_cp550', 't61_cp395_pr4', 'cProp']
    __test_Kfold   =  {   1: ['cRets'     , 't31_cv'           , 't61_crestar_taucprstar', 'supersonic_m3.0r200'   ,  'supersonic_m4.0r200'],
                          2: ['cRets'     , 't31_srestar_taull', 't61_glcprstar'         , 'supersonic_m1.7r200'   ,  'supersonic_m4.0r200',  'jimenez_re950' ],
                          3: ['cRets'     , 'gasLike'          , 'liquidLike'            , 'supersonic_m3.0r600'   ,  'supersonic_m4.0r200',  'jimenez_re550' ],
                          4: ['cRets'     , 't31_ll2'          , 't31_cp150'             , 't61_vlambdasprstar_ll' ,  'supersonic_m3.0r200',  'jimenez_re950' ],
                          5: ['liquidLike', 't31_ll1'          , 't31_cv'                , 't31_cp395'         ,  'supersonic_m0.7r400',  'jimenez_re550' ],
                          6: ['gasLike'   , 't31_cv'           , 't61_crestar_taucprstar', 'supersonic_m0.7r400'   ,  'supersonic_m4.0r200',  'jimenez_re2000'],
                          7: ['cRets'     , 't31_gl'           , 't61_crestar_taucprstar', 't61_glcprstar'         ,  'supersonic_m1.7r200',  'jimenez_re550' ],
                          8: ['gasLike'   , 't31_srestar_taugl', 't31_srestar_taull'     , 't61_vlambdasprstar_ll' ,  'supersonic_m1.7r400',  'jimenez_re4200'],
                          9: ['liquidLike', 't31_cp150'        , 't61_glcprstar'         , 't31_cp395'         ,  'supersonic_m1.7r600',  'jimenez_re180' ],
                         10: ['cRets'     , 't31_srestar_taucv', 't61_crestar_taucprstar', 't31_cp395'         ,  'supersonic_m0.7r400',  'jimenez_re950' ],
                        
                        # 901: ['cRets', 'gasLike', 'liquidLike', 'cProp'],
                        # 902: ['cRets', 'gasLike', 'liquidLike', 'cProp'],
                        # 903: ['cRets', 'gasLike', 'liquidLike', 'cProp'],
                        # 904: ['cRets', 'gasLike', 'liquidLike', 'cProp'],
                        }
    
    __valid_Kfold  =  { # 901: ['Supersonic_M0.7R600', 'Supersonic_M3.0R200', 'T31_LL2'  , 'T31_SReStar_tauCv', 't31_cp395'         , 'T61_GLCPrStar'        , 'Jimenez_Re180', 'Jimenez_Re4200'],
                        # 902: ['Supersonic_M0.7R400', 'Supersonic_M1.7R600', 'T31_CP395', 'T31_CReStar_tau'  , 'T61_CReStar_tauCPrStar', 'T61_GLCPrStar'        , 'Jimenez_Re180', 'Jimenez_Re550' ],
                        # 903: ['Supersonic_M3.0R400', 'Supersonic_M4.0R200', 'T31_CP150', 'jimenez_re550'        , 'T61_GLCPrStar'         , 'T61_VLambdaSPrStar_LL', 'Jimenez_Re180', 'Jimenez_Re2000'],
                        # 904: ['Supersonic_M1.7R400', 'Supersonic_M3.0R600', 'T31_LL1'  , 'T31_SReStar_tauLL', 'T61_CReStar_tauCPrStar', 'T61_VLambdaSPrStar_LL', 'Jimenez_Re950', 'Jimenez_Re2000'],
                        }
    
    __all_cases    =  Import_dns_data.get()['all_cases']
    
    @staticmethod
    def get(Kfold):
        
        banned     =  deepcopy(Build_Kfold.__banned)
        test       =  deepcopy(Build_Kfold.__test_Kfold [Kfold])
        valid      =  deepcopy(Build_Kfold.__valid_Kfold.get(Kfold, []))

        all_cases  =  deepcopy(Build_Kfold.__all_cases)

        
        Build_Kfold.__fix_casenames(banned   )
        Build_Kfold.__fix_casenames(test     )
        Build_Kfold.__fix_casenames(valid    )
        Build_Kfold.__fix_casenames(all_cases)

        train  =  lfilter(lambda c: not (c in set(test+valid+banned)),
                          all_cases                                  )
        
        assert not (set(train) & set(valid))
        assert not (set(train) & set(test ))
        assert not (set(valid) & set(test ))

        assert all(map(lambda dset: all((not c in banned) for c in dset), [train,valid,test]))
        
        return {'train': train,
                'valid': valid,
                'test' : test }
    
    @staticmethod
    def __fix_casenames(A):
        
        ref_cases  =  {b.lower(): b  for b in Build_Kfold.__all_cases}
        assert len(ref_cases) == len(Build_Kfold.__all_cases)

        for i,c in enumerate(A):
            A[i] = ref_cases[c.lower()]

# ------------------------------------------------------------------------
#                  Write Loss Function
# ------------------------------------------------------------------------

class MK_Loss_Function:
    @staticmethod
    def get(delta_loss_type, reg_types):
        
        result = MK_Loss_Function.__write_term('delta', delta_loss_type, 1, apply_other = False)
        
        for field, text_reg in reg_types.items():
            if text_reg is not None:
                result  +=  MK_Loss_Function.__parse_reg(field, text_reg)
        
        return 'lambda delta, other: ' + ' + '.join(result)
    
    @staticmethod
    def __parse_reg(field, text_reg):
        
        text_reg  =  text_reg.split('__')
        assert str(text_reg).count('_') == 0

        result = []
        for i,s in enumerate(text_reg):
            assert s.count('_')==0
            mode, val  =  s.split('x')
            assert mode in ('L1','L2')
            result += MK_Loss_Function.__write_term(field, mode, float(val))
        
        return result


    @staticmethod
    def __write_term(field, mode, val, apply_other = True):
        result   =  "" if abs(val-1)<1e-10 else f"{val}*" #  val == 1 is not worth writing
        f        =  lambda s: {'L1': f'{s}.abs()',
                               'L2': f'{s}**2'   }[mode]
        if apply_other:
            value    =  f"other['{field}']"
            result  +=  f"sum(({f('p')}).sum() for p in {value})/sum(p.numel() for p in {value})"
        else:
            result  +=  f"torch.mean({f(field)})"
        return [result]

# ------------------------------------------------------------------------
#                  Main Runner
# ------------------------------------------------------------------------

def main_runner():

    device, Kfold, param_groups, layers, delta_loss_type, p_loglayer_reg, p_change_reg  =  argv_full  =  argv[1:] + ['']*(8-len(argv))
    argv_full        =  argv_full[1:]

    Kfold            =  int(lstrip_sure(Kfold          , 'Kfold_'  ))
    param_groups     =  int(lstrip_sure(param_groups   , 'pgroups_'))
    # layers
    delta_loss_type  =      lstrip_sure(delta_loss_type, 'delta_loss_' )
    p_loglayer_reg   =      lstrip_sure(p_loglayer_reg , 'p_loglayer_' )  if p_loglayer_reg   else  None
    p_change_reg     =      lstrip_sure(p_change_reg   , 'p_change_'   )  if p_change_reg     else  None

    if p_loglayer_reg == 'None': p_loglayer_reg = None
    if p_change_reg   == 'None': p_change_reg   = None

    output_folder  =  pjoin(  deepdirname(abspath(__file__),3)                                                    ,
                             'data'                                                                               ,
                             'output'                                                                             ,
                             'nn_training'                                                                        ,
                             'production_runs' if Kfold<900 else 'genetic_runs'                                   ,
                             f"run_K_{Kfold}_pgroups_{param_groups}_layers_{layers}_delta_{delta_loss_type}_logreg_{p_loglayer_reg}_mainreg_{p_change_reg}" )

    args = {'output_folder'  :  output_folder                                                          ,
            'argv_full'      :  argv_full                                                              ,
            'device'         :  device                                                                 ,
            'args_net'       :  {'param_groups': param_groups, 'all_layers': parse_layers(layers)}     ,
            'loss_function'  :  MK_Loss_Function.get(delta_loss_type, {'p_loglayer':  p_loglayer_reg,
                                                                       'p_change'  :  p_change_reg  }),
            'lr_w'           :  1e-3                                                                   ,
            'betas_w'        :  (0.9, 0.999)                                                           ,
            'weight_decay_w' :  0.                                                                     ,
            'eps'            :  1e-8                                                                   ,
            'optim_w_type'   :  'adam'                                                                 ,
            'used_cases'     :  Build_Kfold.get(Kfold)                                                 ,
            'epoch_report'   :  100                                                                    ,
            'epoch_predict'  :  100                                                                    ,
            'min_saver_dt'   :    1                                                                    ,
            'epoch_save_freq':  500                                                                    ,
            'epoch_n_saves'  :  3                                                                      ,
            'seeder'         :  {'epochs_avg'     : 100             ,
                                 'metric_name'    : '|diff|/|total|',
                                 'use_dsets'      : ['train']       ,
                                 'epochs_per_seed': 1000            ,
                                 'n_seeds_try'    :  20             }                                  }
    
    args['Epochs']  =  3*15000  +  args['seeder']['epochs_per_seed']*(args['seeder']['n_seeds_try']-1)

    assert args['args_net']['all_layers'  ] == [['T']*5, ['T']*2, ['T']]
    assert args['args_net']['param_groups'] == 3

    assert (args['seeder']['epochs_per_seed'] % args['epoch_save_freq']) == 0
    assert args['epoch_report'] == args['epoch_predict']

    if not isdir(args['output_folder']):
        ML_Runner(args)

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    main_runner()