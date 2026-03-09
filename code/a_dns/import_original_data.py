# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os.path    import  abspath, dirname, isfile
from    sys        import  path                             as  sys_path
from    os.path    import  join                             as  pjoin
from    math       import  isnan                            as  math_isnan
from    importlib  import  machinery                        as  importlib_machinery
from    importlib  import  util                             as  importlib_util     
import  numpy      as      np

# ------------------------------------------------------------------------
#                  Disable Prints (for a moment)
# ------------------------------------------------------------------------

import  contextlib
from    os          import  devnull  as  os_devnull

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(dirname(abspath(__file__))))
from  misc.utils  import  (improved_pformat    ,
                           deepdirname         ,
                           listdir_full_files  ,
                           reader              )
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

array           =  np.array

fname_database  =  pjoin( deepdirname(abspath(__file__),3),
                         'data'                           ,
                         'input'                          ,
                         'dns_data'                       ,
                         'database_dns_cases.pyjson'      )

# ------------------------------------------------------------------------
#                  Utility Functions
# ------------------------------------------------------------------------

def get_unique(ab, key, allow_missing=False):
    a,b = ab
    
    if not allow_missing:
        a,b = a[key], b[key]
    else:
        val_a = [a[key]] if key in a else []
        val_b = [b[key]] if key in b else []
        a     = (val_a or val_b)[0]
        b     = (val_b or val_a)[0]
    
    if type(a) == str:
        assert a == b
    else:
        assert np.fabs(array(a)-array(b)).max() < 1e-10
    return a

# ------------------------------------------------------------------------
#                  Read Original Data
# ------------------------------------------------------------------------

class Internal_original_dns_data_reader:
    @staticmethod
    def get():
        # returns dictionary with original (matlab) data
        #     -> cases are unsorted yet
        
        result  =  {}
        for mode in ['kinematic', 'thermal']:
            for fname in listdir_full_files(pjoin(dirname(fname_database), 'original_data', f'{mode}_data')):
                data          =  Internal_original_dns_data_reader.__mk_dict(fname)
                data['fname'] =  fname
                case          =  data['Casename']
                
                if not case in result:  result[case] = {}
                
                assert not mode in result[case]
                result[case][mode] = data
        
        return result
    
    @staticmethod
    def __mk_dict(fname):
        
        # import data as dictionary
        with open(os_devnull, "w") as f, contextlib.redirect_stdout(f):
            loader = importlib_machinery.SourceFileLoader('getMatlabData', fname)
            spec   = importlib_util.spec_from_loader(loader.name, loader)
            mod    = importlib_util.module_from_spec(spec)
            loader.exec_module(mod)
            data   = mod.getMatlabData().__dict__
        
        # process arrays and nan
        for key,val in data.items():
            if hasattr(val,'shape'):
                if np.prod(val.shape) == 1:
                    data[key]  =  float(val.flatten()[0])
                else:
                    data[key]  =  val.tolist()
            elif (not type(val) in [str,list]) and math_isnan(val):
                data[key] = float('nan')
        return data

# ------------------------------------------------------------------------
#                  Import DNS Data (ready for usage)
# ------------------------------------------------------------------------

class Import_dns_data:
    @staticmethod
    def get():
        if not isfile(fname_database):  Import_dns_data.build()
        return eval(' '.join(reader(fname_database)))
    
    @staticmethod
    def build():
        original_data  =  Internal_original_dns_data_reader.get() 
        result         = {'all_cases':  Import_dns_data.__sort_cases(original_data.keys()),
                          'data'     :  {}                                                }
        for case in result['all_cases']:
            result['data'][case] = Import_dns_data.__mk_fusion(original_data[case])
        
        with open(fname_database, 'w') as f:
            f.write(improved_pformat(result) + '\n')
    
    @staticmethod
    def __mk_fusion(d):
        assert sorted(d.keys()) == ['kinematic', 'thermal']
        k, t  = kt =  d['kinematic'], d['thermal']
        
        result      =  {'case'          :  get_unique(kt, 'Casename')                     ,
                        'u_dns'         :  k['u_DNS']                                     , # get_unique(kt, 'u_dns'   ),
                        'y'             :  get_unique(kt , 'y'      )                     ,
                        'Ret'           :  get_unique(kt , 'ReT'    )                     ,
                        'original_files': {'kinematic': k['fname'], 'thermal': t['fname']},
                        }
        
        bool_is_JI = 'jimenez'    in result['case'].lower()
        bool_is_TL = 'supersonic' in result['case'].lower()
        bool_is_AP = (('t31_' in result['case'].lower()) or 
                      ('t61_' in result['case'].lower()) or 
                      (result['case'].lower() in ['cprop', 'crets', 'gaslike', 'liquidlike']))
        
        assert (bool_is_JI + bool_is_TL + bool_is_AP) == 1
        
        if bool_is_JI:
            result['Pr']     =  1.
            result['cp']     =  1.
            result['T_dns']  =  [1. for _ in result['y']]
        else:
            result['Pr']     =  get_unique(kt , 'Pr'    , allow_missing = True)
            result['cp']     =  get_unique(kt , 'cp_gas', allow_missing = True)
            result['T_dns']  =  get_unique(kt , 'T_DNS' , allow_missing = True)
        
        mu_wall              =  1./result['Ret']
        result['A_sca_r'  ]  =  1.
        result['A_sca_mu' ]  =  mu_wall
        result['A_sca_k'  ]  =  result['cp']*mu_wall/result['Pr'] # k_wall
        
        if bool_is_JI:
            result['b_exp_r'    ]  =  0.
            result['b_exp_mu'   ]  =  0.
            result['b_exp_k'    ]  =  0.
            result['source_heat']  =  0.
            result['visc_heat'  ]  =  False
        
        elif bool_is_AP:
            result['b_exp_r'  ]     =  t['AP_exp_r']
            result['b_exp_mu' ]     =  t['AP_exp_mu']
            result['b_exp_k'  ]     =  t['AP_exp_lamd']
            result['source_heat' ]  =  t['AP_Qt']/(result['Ret']*result['Pr'])
            result['visc_heat']     =  False
        
        elif bool_is_TL:
            assert abs(t['TL_prod_Tr']  - 1      ) < 1e-10
            assert abs(t['TL_scale_mu'] - mu_wall) < 1e-10
            
            result['b_exp_r'  ]     =  -1.
            result['b_exp_mu' ]     =  t['TL_exp_mu']
            result['b_exp_k'  ]     =  result['b_exp_mu' ]
            result['source_heat' ]  =  0.
            result['visc_heat']     =  True
        
        if    abs(result['b_exp_mu']) > 1e-2:
            mu_data = array(k['mu'])
            assert abs(mu_data[0] - mu_wall) < 1e-10
            result['T_dns'] = (mu_data/mu_data[0])**(1./result['b_exp_mu' ])
            
        elif  abs(result['b_exp_r' ]) > 1e-2:
            r_data = k['r']
            assert abs(r_data[0]-1) < 1e-10
            result['T_dns'] = (r_data/r_data[0])**(1./result['b_exp_r' ])
        
        # assert np.fabs(array(result['T_dns'])-ref_T_dns).max() < 1e-3,(array(result['T_dns'])-ref_T_dns,result['T_dns'],ref_T_dns)
        
        result['rho_dns'] = (result['A_sca_r' ]*(np.array(result['T_dns'])**result['b_exp_r' ])).tolist()
        result['mu_dns' ] = (result['A_sca_mu']*(np.array(result['T_dns'])**result['b_exp_mu'])).tolist()
        result['k_dns'  ] = (result['A_sca_k' ]*(np.array(result['T_dns'])**result['b_exp_k' ])).tolist()

        for key,val in result.items():
            if hasattr(val,'shape'):
                result[key] = val.tolist()
        
        return result
        
    @staticmethod
    def __sort_cases(cases):
        cases           =  list(cases)
        expected_cases  =  ['cProp', 'cRets', 'gasLike', 'liquidLike', 'Supersonic_M0.7R400', 'Supersonic_M0.7R600', 'Supersonic_M1.7R200', 'Supersonic_M1.7R400', 'Supersonic_M1.7R600', 'Supersonic_M3.0R200', 'Supersonic_M3.0R400', 'Supersonic_M3.0R600', 'Supersonic_M4.0R200', 'T31_CP150', 'T31_CP395', 'T31_CP550', 'T31_CReStar_tau', 'T31_Cv', 'T31_GL', 'T31_LL1', 'T31_LL2', 'T31_SReStar_tauCv', 'T31_SReStar_tauGL', 'T31_SReStar_tauLL', 'T61_CP395_Pr4', 'T61_CReStar_tauCPrStar', 'T61_GLCPrStar', 'T61_VLambdaSPrStar_LL', 'Jimenez_Re180', 'Jimenez_Re550', 'Jimenez_Re950', 'Jimenez_Re2000', 'Jimenez_Re4200']
        assert sorted(cases) == sorted(expected_cases)
        return expected_cases


# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    imported_data = Import_dns_data.get()