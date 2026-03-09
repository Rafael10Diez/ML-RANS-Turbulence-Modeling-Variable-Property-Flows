
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from  os.path     import  abspath, isdir, isfile, dirname, basename
from  os          import  listdir                                    as  os_listdir
from  os.path     import  join                                       as  pjoin
from  sys         import  platform                                   as  sys_platform
from  sys         import  argv
from  copy        import  deepcopy
from  collections import  Counter
from  os          import  system as os_system
import random

random.seed(0)

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

def get_folder(tag):
    x = dirname(abspath(__file__))
    while not tag in basename(x):
        x = dirname(x)
    return x

def pop1(A):
    x, = A
    return x

def reader(fname):
    with open(fname,'r') as f:
        return [x.rstrip('\n') for x in f]

deepdirname           =  lambda x,n:  x if n<1 else deepdirname(dirname(x), n-1)

lfilter               =  lambda f,x:  list(filter(f,x))
lmap                  =  lambda f,x:  list(map   (f,x))
listdir_full          =  lambda   x:  sorted(pjoin(x,y) for y in os_listdir(x))

listdir_full_folders  =  lambda   x:  lfilter(isdir , listdir_full(x))
listdir_full_files    =  lambda   x:  lfilter(isfile, listdir_full(x))

get_avg               =  lambda arr: sum(arr)/len(arr)

def lstrip_check(x, tag):
    assert (x[:len(tag)] == tag) and (x.count(tag)==1)
    return x.lstrip(tag)

def sorted_dict(D):
    return {k:v for k,v in sorted(D.items(), key = lambda kv: kv[1])}

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def fmt_layer(layer):
    assert type(layer) == list
    if layer:
        assert type(layer[0]) == str
    return str(sorted(layer))

def fmt_arch(arch):
    assert sorted(arch.keys()) == ['all_layers', 'param_groups']
    assert (type(arch['all_layers'  ]) == list) and \
           (type(arch['param_groups']) == int )
    return str({ 'all_layers'  :  list(map(lambda x: eval(fmt_layer(x)), arch['all_layers'  ])) ,
                 'param_groups':                                     int(arch['param_groups'])  })

# ------------------------------------------------------------------------
#                  Scan Folder
# ------------------------------------------------------------------------

class Scan_folder:
    def __init__(self, folder):
        self.folder     =  folder
        self.log_fname  =  pop1(lfilter(lambda x: x.endswith('.log'),
                                        listdir_full_files(folder)  ))
        
        self.__A_log            =  reader(self.log_fname)
        self.runner_args        =  self.__scan_dict('# Begin_Input_Arguments' , '# End_Input_Arguments')
        self.finished_training  =  len(lfilter(lambda x: x[:19] == '# Finished_Training', self.__A_log)) == 1

        self.kfold              =  int(lstrip_check(self.runner_args['argv_full'][0], 'Kfold_'))
        self.key_arch           =  tuple(self.runner_args['argv_full'][1:])

        if self.finished_training:
            self.pred_stats        =  self.__scan_dict('# Begin_Pred_Stats'      , '# End_Pred_Stats')
            self.__find_ml_error()
    
    def __scan_dict(self, begin, end):
        A                 =  self.__A_log
        i_begin           =  pop1(lfilter(lambda i: A[i][:len(begin)] == begin, range(len(A))))
        i_end             =  pop1(lfilter(lambda i: A[i][:len(end)  ] == end  , range(len(A))))
        return eval(' '.join(A[i_begin+1:i_end]))
    
    def __find_ml_error(self                       ,
                        n_epochs = 500             ,
                        all_dset =  ['valid']      ,
                        metric   = '|diff|/|total|'):
        assert  self.finished_training
        assert  not  'test'  in  all_dset
        result  =  []
        for dset in all_dset:
            info       =  self.pred_stats[dset]
            threshold  =  info[-1]['epoch'] - n_epochs
            i, found   =  -1, []
            while info[i]['epoch'] >= threshold:
                found.append(info[i][metric])
                i -= 1
            result.append(get_avg(found))
        self.ml_error  =  get_avg(result)

# ------------------------------------------------------------------------
#                  Scan architectures (group folders)
# ------------------------------------------------------------------------

class Group_Folders:
    def __init__(self, args):
        self.args           =  args
        self.wished_kfolds  =  args['wished_kfolds']
        self.isvalid        =  args['isvalid']

        self.best_arch      =  dict(score = float('inf'))
        self.seen_archs     =  set()
        self.found          =  {}
        self.full_score     =  {}

        for m in lmap(Scan_folder, listdir_full_folders(pjoin( deepdirname(abspath(__file__),3),
                                                              'data'                           ,
                                                              'output'                         ,
                                                              'nn_training'                    ))):
            
            if (m.kfold in self.wished_kfolds) and self.isvalid(m.key_arch):
                
                self.seen_archs.add(fmt_arch(m.runner_args['args_net']))

                if m.finished_training:
                    
                    if not m.key_arch in self.found:  self.found[m.key_arch] = {}

                    self.found[m.key_arch][m.kfold] = m

                    if len(self.found[m.key_arch]) == len(self.wished_kfolds):
                        self.full_score[m.key_arch]  =  score  =  get_avg(lmap(lambda obj: obj.ml_error       ,
                                                                               self.found[m.key_arch].values()))
                        if score < self.best_arch['score']:
                            self.best_arch['score']    = score
                            self.best_arch['key_arch'] = m.key_arch
        
        self.full_score = sorted_dict(self.full_score)
        self.print_ranking()

    def print_ranking(self):
        max_len_key  =  max(map(lambda k: len(str(k)), self.full_score.keys()))
        fmt_key      =  eval("lambda x: f'{str(x):" + str(max_len_key) + "}'")
        i            =  0
        print('\n\nRanking Architectures:')
        for key_arch, score in self.full_score.items():
            i  +=  1
            print(f'Rank: {i:3d}    (score = {(score*100):.6f} %)    {fmt_key(list(key_arch))}')
        print('')

# ------------------------------------------------------------------------
#                  Mutate Architectures
# ------------------------------------------------------------------------

class Gen_Mutations:
    def __init__(self, parent, device, last_argv):
        self.parent       =  parent
        self.best_params  =  pop1(list(set(map(lambda m: fmt_arch(m.runner_args['args_net'])      ,
                                               parent.found[parent.best_arch['key_arch']].values()))))
        self.mutations    =  set()

        self.mutate_param_groups()
        self.mutate_layers()
        self.add_layers()

        for str_arch in parent.seen_archs:
            if str_arch in self.mutations:
                self.mutations.remove(str_arch)
        
        self.mutations = sorted(self.mutations)
        self.mk_queue_argv(device, last_argv)
    
    def mutate_param_groups(self):
        for d in [-1,1]:
            params                   =  eval(self.best_params)
            params['param_groups']  +=  d
            self.mutations.add(fmt_arch(params))
    
    def mutate_layers(self):
        
        for i, layer in enumerate(eval(self.best_params)['all_layers']):
            
            alternatives = {fmt_layer(layer),}

            for j in range(len(layer)):
                new = layer[:j] + layer[j+1:]
                if new:
                    alternatives.add(fmt_layer(new))
            
            for new in deepcopy(alternatives):
                for extra in [['T'],['S'],['R'],['P'],[]]:
                    alternatives.add(fmt_layer(eval(new) + extra))
            
            for new in alternatives:
                params                   =  eval(self.best_params)
                params['all_layers'][i]  =  eval(new)
                self.mutations.add(fmt_arch(params))
    
    def add_layers(self):
        all_layers = eval(self.best_params)['all_layers']
        for extra in 'TSRP':
            for i in range(len(all_layers)+1):
                new = deepcopy(all_layers)
                new.insert(i,[extra])
                params               = eval(self.best_params)
                params['all_layers'] = new
                self.mutations.add(fmt_arch(params))
    
    def mk_queue_argv(self, device, last_argv):
        self.queue_argv = []
        
        for arch in self.mutations:
            arch = eval(arch)

            all_layers  =  arch['all_layers']
            str_layers  =  []

            for layer in all_layers:
                C = Counter(layer)
                str_layers.append('.'.join(f"{k}{C[k]}" for k in sorted(C.keys())))
            
            str_layers  =  '__'.join(str_layers)
            self.queue_argv.append([device, 'Kfold_PYTHON_TAG_KFOLD', f"pgroups_{str(arch['param_groups'])}", str_layers, *last_argv])
        random.shuffle(self.queue_argv)



# ------------------------------------------------------------------------
#                  Main Architecture
# ------------------------------------------------------------------------

def main_runner(device):
    if 'win' in sys_platform:
        python_path     =  f'{pjoin(get_folder("rafae")        , "anaconda3","Scripts","activate.bat")} && python'
    else:
        python_path     =  'python'
    cache_abspath       =  pjoin(dirname(dirname(dirname(abspath(__file__)))), "clean_cache.py")
    main_file           =  pjoin(dirname(abspath(__file__)),'main.py')

    while True:
        last_argv     =  'delta_loss_L2', '', ''
        wished_kfolds =  [901, 902, 903, 904]

        g           =  Group_Folders(dict(wished_kfolds = [901, 902, 903, 904]                       ,
                                          isvalid       = lambda key_arch: key_arch[-3:] == last_argv))
        queue_argv  =  Gen_Mutations(g, device, last_argv).queue_argv
        line = '-'*20
        print(f'\n{line} Number of Mutations: {len(queue_argv)} {line}\n')
        for new_argv in  queue_argv:
            print(new_argv)

if __name__ == '__main__':
    device = 'cuda'
    main_runner(device)