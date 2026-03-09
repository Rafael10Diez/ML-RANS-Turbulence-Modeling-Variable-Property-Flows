# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  zipfile
from    os.path      import  join                                       as  pjoin
from    os           import  listdir                                    as  os_listdir
from    os.path      import  basename, abspath, dirname, isfile, isdir

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

deepdirname           =  lambda x,n:  x if n<1 else deepdirname(dirname(x), n-1)

lfilter               =  lambda f,x:  list(filter(f,x))
lmap                  =  lambda f,x:  list(map   (f,x))

listdir_full          =  lambda x  :  sorted(pjoin(x,y) for y in os_listdir(x))
listdir_full_files    =  lambda x  :  lfilter(isfile, listdir_full(x))
listdir_full_folders  =  lambda x  :  lfilter(isdir , listdir_full(x))

def reader_zip(saved_zip):
    assert  isfile(saved_zip)
    local_name  =  basename(saved_zip).removesuffix('.zip')
    return zipfile.Path(saved_zip, at = local_name).read_text().split('\n')

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

class Read_FI_Data:
    @staticmethod
    def get(used_cases):
        
        all_data   =  Read_FI_Data._get_all_folders()
        result     =  {} # all_D

        for dset in ['train','valid','test']:
            
            result[dset] = {}
            
            for case in used_cases[dset]:
                result[dset][case] = {'ini':  Read_FI_Data.__fetch_ini(all_data[case]['ini']),
                                      'end':  Read_FI_Data.__fetch_end(all_data[case]['end'])}
        return result
    
    @staticmethod
    def _get_all_folders(b_mode = 'Beta_k', u_imp = 100, field_inversion_folder = 'field_inversion'):
        # this function returns all data belonging to all field inversion runs
        # dictionary is not sorted in order

        # all folders
        all_folders =  lfilter(lambda x: 'FI_run' == basename(x)[:6]                                         ,
                               listdir_full_folders(pjoin(deepdirname(abspath(__file__),3), 'data', 'output', field_inversion_folder, b_mode, f'u_imp_{u_imp}')))
        
        assert len(all_folders) == {'Beta_k'                  : 33,
                                    'Beta_e'                  : 33,
                                    'Beta_k__1__Beta_e__1'    : 33,
                                    'Beta_k__0.5__Beta_e__1.5':  1,
                                    'Beta_k__1.5__Beta_e__0.5':  1}[b_mode]
        
        # fetch results
        result = {}
        for     folder      in all_folders:
            for tag, i_pick in [['ini',  0],
                                ['end', -1]]:
                saved_zip          =  sorted(filter(lambda x: x.endswith('.dat.zip'), listdir_full_files(folder)),
                                                    key = lambda x: int(x.split('_state_iter_')[1].split('.')[0]))[i_pick] 
                
                if tag == 'ini':  assert saved_zip.endswith('_state_iter_0.dat.zip')
                inf                =  float('inf')
                d                  =  eval(' '.join(reader_zip(saved_zip)))
                case               =  d['args']['args_mk_solver']['case']
                
                if not case in result:  result[case] = {}
                
                result[case][tag]  =  d
                if (tag == 'end') and ((not d.get('finished_running',False)) and d['iters']<15000):
                    print(f'Early Finish: {basename(saved_zip)}',d.get('finished_running',False))
        return result
    
    @staticmethod
    def __fetch_ini(D):
        cfd_keys   =  ['rans_u'        , 'rans_MK_k'    , 'rans_MK_e'   ,
                       'rans_rho_molec', 'rans_mu_molec', 'rans_mu_turb',
                       'Ret'           , 'rans_MK_sig_k', 'y'           ]
        hard_keys  =  ['Ru_basis'      , 'Rk_basis'     , 'Re_basis'    ]
        geom_keys  =  [  'i_s',   'i_c',   'i_n' ,
                       'cd1_s', 'cd1_c', 'cd1_n' , 
                       'cd2_s', 'cd2_c', 'cd2_n' ]
        result               =  {key: D['cfd_solver'][key]               for key in cfd_keys }
        result.update(          {key: D['hard_code' ][key]               for key in hard_keys})
        result['geom_vars']  =  {key: D['cfd_solver']['geom_vars'][key]  for key in geom_keys}
        return result

    @staticmethod
    def __fetch_end(D):
        cfd_keys   =  ['rans_rho_molec' , 'rans_MK_e' , 'beta_distrib_K' ]
        hard_keys  =  ['Rk_basis'       ]
        result     =  {key: D['cfd_solver'][key] for key in cfd_keys }
        result.update({key: D['hard_code' ][key] for key in hard_keys})
        return result
