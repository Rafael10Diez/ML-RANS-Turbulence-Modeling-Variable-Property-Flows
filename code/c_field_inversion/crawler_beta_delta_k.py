print('    WARNING:    this is only a configuration script')
# ------------------------------------------------------------------------
#                  Generic libraries
# ------------------------------------------------------------------------

from    os.path            import  isfile, isdir, abspath, basename, dirname
from    pprint             import  pformat
from    os.path            import  join                                       as  pjoin
from    os                 import  listdir                                    as  os_listdir
from    sys                import  path                                       as  sys_path
import  numpy              as      np
from    copy               import  deepcopy

# ------------------------------------------------------------------------
#                  Custom libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(dirname(abspath(__file__))))
from a_dns.import_original_data             import  Import_dns_data
from misc.utils                             import  (reader_zipped_text, 
                                                     lmap, 
                                                     lfilter,
                                                     pop1,
                                                     deepdirname,
                                                     listdir_full_files, 
                                                     listdir_full_folders, 
                                                     reader)
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic definitions
# ------------------------------------------------------------------------

array                 =  np.array
lmap                  =  lambda f,x: list(map   (f,x))
lfilter               =  lambda f,x: list(filter(f,x))
listdir_full          =  lambda x: [pjoin(x,y) for y in os_listdir(x)]
listdir_full_files    =  lambda x: lfilter(isfile, listdir_full(x))
listdir_full_folders  =  lambda x: lfilter(isdir , listdir_full(x))
apply_pformat         =  lambda D: pformat(D, sort_dicts=False, indent = 2)

power                 =  lambda x,y: x**y

def clean_num(x):
    if '.' in x:
        x = x.rstrip('0')
        if x[-1] == '.':
            x = x[:-1]
    return eval(x)

def add_value(result, all_key, value):
    d = result
    for i,key in enumerate(all_key):
        if not key in d:
            if i == len(all_key)-1:
                d[key] = value
            else:
                d[key] = {}
        d      = d[key]

# ------------------------------------------------------------------------
#                  Read Data
# ------------------------------------------------------------------------


all_cases = Import_dns_data.get()['all_cases']

# ------------------------------------------------------------------------
#                  Align dictionaries
# ------------------------------------------------------------------------

def super_align(x):
    if type(x) == dict:
        was_str  = type(list(x.keys())[0]) == str
        if was_str:
            keys     =  lmap(lambda k: f"'{k}'", x.keys())  
        else:
            keys     =  lmap(str, x.keys())
        values   =  lmap(super_align, x.values())
        max_len  =  max(map(len, keys))
        fmt      =  eval('lambda x: f"{x:' + str(max_len) + 's}: "')
        keys     =  list(map(fmt, keys))
        max_len  =  max(map(len, keys))
        A        = []
        for key,val in zip(keys,values):
            A.append(key + val[0])
            for i in range(1,len(val)):
                A.append(' '*max_len + val[i])
            A[-1] += ','
        A[0] = '{' + A[0]
        for i in range(1,len(A)):
            A[i] = ' ' + A[i]
        A[-1] = A[-1][:-1] + '}'
        return A
    else:
        return [str(x)]

# ------------------------------------------------------------------------
#                  Fetch Variables
# ------------------------------------------------------------------------

def conv_var_brackets(x):
    if x.count('[') == x.count(']') == 1:
        return x[x.index('[') : (x.index(']')+1)]
    else:
        if '=' in x:
            x = x.split('=')[1]
        return x.replace(' ','').rstrip('\n')

def conv_var_log(x):
    x  =  conv_var_brackets(x).replace(' ' , ',').replace('[,' , '[').replace(',]' , ']')
    # assert (x[0] == '[') and (x[-1] == ']')
    while ',,' in x:
        x = x.replace(',,' , ',')
        x = x.replace('[,' , '[').replace(',]' , ']')
    return eval(x)

def fetch_var(A, tag, conv, offset):
    i      =  lfilter(lambda i: A[i][:len(tag)] == tag, 
                      range(len(A))                   )[-1]
    return conv(A[i+offset])


# ------------------------------------------------------------------------
#                  Process subfolder
# ------------------------------------------------------------------------

def process_subfolder(folder, typerun, result, D_all_cases):
    
    locate    =  lambda end, folder: pop1(lfilter(lambda x: x.endswith(end) ,
                                                  listdir_full_files(folder)))
    
    A_log     =  reader(locate('.log', folder))

    if any(map(lambda x: x.endswith('.py'), listdir_full_files(folder))):
        A_matlab  =  reader(locate('.py' , folder))
    else:
        A_matlab  =  reader(locate('.py' , dirname(folder)))
    
    data = {}
    for key in ['u','k','e','Rk_basis','Re_basis','betaDistrib_K','betaDistrib_E']:
        prop  =  'Optim' if key in ['Rk_basis','Re_basis'] else 'FieldVars'
        try:
            data[key]  =  fetch_var(A_log, f'MESH.{prop}.{key}', conv_var_log, 1)
        except:
            assert key in ['betaDistrib_K','betaDistrib_E']
            data[key]  =  [1. for _ in data['u']]
    
    dot_conv  =  lambda x: x.split(':')[1].replace(' ','').lower()

    case      =  D_all_cases[fetch_var( A_log                    ,
                                       'casename              :' ,
                                        dot_conv                 ,
                                        0                        )]
    
    u_imp     =  fetch_var( A_log                            ,
                           'u_importance          :'         ,
                            lambda x: clean_num(dot_conv(x)) ,
                            0                                )
    
    for key in ['mu', 'r', 'y', 'Ce2', 'ReT', 'u_DNS']:
        data[key] = fetch_var(A_matlab, f'    MatlabData.{key} ', lambda x: eval(conv_var_brackets(x)), 0)
    
    for key in data:
        data[key] = np.array(data[key])

    d               =  data['y'  ] 
    mu              =  data['mu' ]
    r               =  data['r'  ]
    k               =  data['k'  ]
    e               =  data['e'  ]
    Ret             =  data['ReT']
    Ce2             =  data['Ce2']

    yplus           =  d*np.sqrt(r/r[0])/(mu/mu[0])*Ret
    ReTurb          =  r*power(k, 2)/(mu*e)
    f2              = (1.-2./9.*np.exp(-power(ReTurb/6., 2)))*power(1-np.exp(-yplus/5.), 2)
    
    data['y_star']  =  yplus  # modified yplus is actually Y*
    data['dest_K']  =  -r*e
    k               =  k + 0.
    k[k<1e-12]      =  1e-12
    data['dest_E']  =  -r*Ce2*f2*(e**2)/k

    for b in 'KE':
        data[f'deltaDistrib_{b}']  =  data[f'dest_{b}'] * (data[f'betaDistrib_{b}'] - 1)
    
    data['u_error']  =  np.fabs(data['u']-data['u_DNS']).max() / np.fabs(data['u_DNS']).max()

    data['mu_DNS'      ]  =  mu
    data['r_DNS'       ]  =  r
    data['u_field_inv' ]  =  data['u']
    data['k_field_inv' ]  =  k
    data['e_field_inv' ]  =  e

    new = {}
    for     b    in 'KE':
        for var_ in ['beta','delta']:
            new[f'{var_}_distrib_{b}']  =  data[f'{var_}Distrib_{b}'].tolist()
    
    for b in 'ke':
        new[f'R{b}_basis']  =  data[f'R{b}_basis']

    for key in ['u_DNS', 'u_error', 'y_star', 'y', 'mu_DNS', 'r_DNS', 'u_field_inv', 'k_field_inv', 'e_field_inv']:
        new[key] = data[key].tolist()

    add_value( result                ,
              [typerun, u_imp, case] ,
              new                    )

# ------------------------------------------------------------------------
#                  Scan root folder
# ------------------------------------------------------------------------

def scan_folder(main_folder, howto, typerun, result):
    
    banned    =  ['Dill_Result_Explorer', 'Dill_Result_Explorer_eimp0', 'core', 'FI_MK_Prt_Optim_uimp100']
    
    D_all_cases  =  {key.lower(): key for key in all_cases}
    assert len(D_all_cases) == len(all_cases)

    if howto:
        
        seen = {case: {u_imp: 0 for u_imp in map(lambda i: 10**i, range(-1,4))} for case in all_cases}
        
        for      case_folder  in  listdir_full_folders(main_folder):
            
            if basename(case_folder) in banned:
                continue
            
            case  =  D_all_cases[basename(case_folder).lower()]

            for  folder       in  listdir_full_folders(case_folder):
                seen[case][eval(folder.split('_')[-1])] += 1
                process_subfolder(folder, typerun, result, D_all_cases)
        
        assert all((v==1) for all_v in seen.values() for v in all_v.values())
    
    else:
        
        process_subfolder(main_folder, typerun, result, D_all_cases)

# ------------------------------------------------------------------------
#                  Compare dictionaries
# ------------------------------------------------------------------------

def compare_results(new, old):
    # new[only_Beta_k][100][cRets][beta_distrib_K], etc.
    max_error = float('-inf')
    for b_mode in new.keys():
        if b_mode == 'Ik_1_Ie_1':
            continue
        
        for u_imp in new[b_mode].keys():
            if u_imp != 100:
                continue

            for         case in new[b_mode][u_imp].keys():
                for     var_ in ['beta','delta']:
                    for bb   in 'KE':

                        temp_new  =  new[b_mode][u_imp][case]
                        temp_old  =  old[f"{var_}_distrib"][b_mode][case]

                        a  =  array(temp_new[f"{var_}_distrib_{bb}"])

                        if bb in temp_old:
                            b = array(temp_old[bb])
                        else:
                            b = {'beta' : np.ones_like ,
                                 'delta': np.zeros_like}[var_](a)
                        
                        ref   = {'beta' : 1                               ,
                                 'delta': temp_new[f"R{bb.lower()}_basis"]}[var_]
                        error =  np.fabs(a-b).max()/ref * 100.
                        max_error = max(max_error, error)
                        assert error < 2
    print('max_error: ', max_error)
    return True
# ------------------------------------------------------------------------
#                  Scan root folder
# ------------------------------------------------------------------------

def build_ref_fi_data(to_folders, target_fname):
    
    result  =  {}
    
    for typerun, howto, folder in to_folders:
        scan_folder(folder, howto, typerun, result)
    
    with open(target_fname, 'w') as f:
        f.write('\n'.join(super_align(result))+'\n')
    
    return result

if __name__ == '__main__':
    root_dir         =  r'C:\Users\rafae\MSc_Thesis_TUD\2017-2018\Thesis'
    to_folders       =  [['only_Beta_k'  , True , pjoin(root_dir, 'Final_Simulations_MK_eimp0'                                          )],
                         ['only_Beta_e'  , True , pjoin(root_dir, 'Final_Simulations_MK_kimp0_v2'                                       )],
                         ['Ik_1_Ie_1'    , True , pjoin(root_dir, 'Final_Simulations_MK'                                                )],
                         ['Ik_0.5_Ie_1.5', False, pjoin(root_dir, 'MK_Final_Sims_Sensitivity_crets_u_imp_100_IkIeChange', 'u100k0.5e1.5')],
                         ['Ik_1.5_Ie_0.5', False, pjoin(root_dir, 'MK_Final_Sims_Sensitivity_crets_u_imp_100_IkIeChange', 'u100k1.5e0.5')]]
    
    target_fname     =  pjoin(deepdirname(abspath(__file__),3), 'data', 'input', 'reference_field_inversion_results.pyjson')
    
    ref_fi_data      =  build_ref_fi_data(to_folders, target_fname)
    old_ref_fi_data  =  eval(' '.join(reader_zipped_text(pjoin(dirname(target_fname), f'old_{basename(target_fname)}'))))

    assert compare_results(ref_fi_data, old_ref_fi_data)



