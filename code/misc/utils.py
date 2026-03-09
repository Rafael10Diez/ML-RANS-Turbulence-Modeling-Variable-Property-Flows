# ----------------------------------------------------------------------------------
#                  Generic Libraries
# ----------------------------------------------------------------------------------

import  socket
import  zipfile
from    os.path      import  basename, dirname, isfile, isdir
from    os.path      import  join                              as  pjoin
from    os           import  listdir                           as  os_listdir
from    collections  import  OrderedDict
from    pprint       import  pformat
from    datetime     import  datetime
from    shutil       import  make_archive

# ----------------------------------------------------------------------------------
#                  Basic Definitions
# ----------------------------------------------------------------------------------

get_hostname      =  lambda: socket.gethostname()
calc_net_params   =  lambda net: sum(p.numel() for p in net.parameters() if p.requires_grad)

fmt_D             =  lambda   D: str(D).replace(' ','')
fmt_p             =  lambda   x: f'{(x*100):.5f} %'
lmap              =  lambda f,x: list(map   (f,x))
lfilter           =  lambda f,x: list(filter(f,x))

deepdirname       =  lambda x,n: x if n<1 else deepdirname(dirname(x),n-1)

listdir_full          =  lambda   x:  sorted(pjoin(x,y) for y in os_listdir(x))
listdir_full_folders  =  lambda   x:  lfilter(isdir , listdir_full(x))
listdir_full_files    =  lambda   x:  lfilter(isfile, listdir_full(x))

def reader(fname):
    with open(fname, 'r') as f:
        return [x.rstrip('\n') for x in f]

def sorted_dict_by_key(D):
    return {k:v for k,v in sorted(D.items(), key = lambda kv: kv[0])}

def sorted_dict_by_val(D):
    return {k:v for k,v in sorted(D.items(), key = lambda kv: kv[1])}

def pop1(A):
    x, = A
    return x

# ----------------------------------------------------------------------------------
#                  Time Stamps
# ----------------------------------------------------------------------------------

tstamp     =  lambda   :  datetime.now().strftime("%Y%m%d_%H%M%S")
full_time  =  lambda   :  datetime.now().strftime("%H:%M:%S, %d/%m/%Y")
format_dt  =  lambda x :  "%02d:%02d:%02d [hh:mm:ss]" % (x//3600, (x%3600)//60, x%60)

# ----------------------------------------------------------------------------------
#                  FPrint
# ----------------------------------------------------------------------------------

class FPrint:
    def __init__(self, fname, track_all = False):
        self.fname      =  fname
        self.fwrite     =  open(fname, 'w')
        self.track_all  =  track_all
        self.__A        =  []
        
    def __call__(self, *line):
        line = self.spacejoin(line)
        print(line)
        self.only_to_file(line)
    
    def only_to_file(self, *line):
        line = self.spacejoin(line)
        self.fwrite.write(line + '\n')
        self.fwrite.flush()
        if self.track_all: self.__A.append(line)
    
    def get_all(self):
        assert self.track_all
        return self.__A
    
    close  =  __del__  =  lambda self: self.fwrite.close()

    @staticmethod
    def spacejoin(x): 
        return ' '.join(map(str,x))

# ----------------------------------------------------------------------------------
#                  Make zipped copy of folder
# ----------------------------------------------------------------------------------

# copy a zipped version of source_folder into: pjoin(target_folder, 'python_code.zip')
def Copy_Zipped(source, target):
    make_archive(  pjoin(target, 'python_code')  ,
                   'zip'                         ,
                   source                        )

# ----------------------------------------------------------------------------------
#                  Improved version of pprint.pformat
# ----------------------------------------------------------------------------------

# improved version of pformat:
#   works with torch.tensor objects
#   does not split lists with newline
def improved_pformat(D):
    assert type(D) == dict
    encode  =  {}
    def dfs(D):
        result = {}
        assert type(D) == dict
        for k,v in D.items():
            if Dictify_obj.is_obj(v):
                result[k] = str(v)
            else:
                if type(v) == dict:
                    result[k] = dfs(v)
                elif (type(v) == list) or hasattr(v,'shape'):
                    if hasattr(v,'shape'): v = v.tolist()
                    new_code  =  f"[code{len(encode):06d}code]"
                    assert not new_code in str(D)
                    encode[new_code] = v
                    result[k] = new_code
                else:
                    result[k] = v
        return result
    D = pformat(dfs(D), sort_dicts=False, indent = 2)
    for code,val in encode.items():
        D = D.replace(f"'{code}'", str(val))
    # eval(str(D))
    return D

# ----------------------------------------------------------------------------------
#                  Improved conversion of Pytorch tensor into string
# ----------------------------------------------------------------------------------

# format a pytorch tensor
# such that x = eval(fmtTensor(x))
# full decimal precision is preserved
class fmtTensor:
    def __init__(self, A, tensor_keyword):
        assert hasattr(A,'shape')
        self.dtype = str(A.dtype)
        self.A     = A.tolist() # convert tensor to list
        assert type(self.A) in [list,int,float]
        assert type(tensor_keyword) == bool
        self.tensor_keyword = tensor_keyword
    def __print(self):
        s = fmt_D(self.A)
        assert type(s) == str
        return f'torch.tensor({s},dtype={self.dtype})' if self.tensor_keyword else s
    __repr__ = __str__ = __print

# ----------------------------------------------------------------------------------
#                  Convert state_dict into string
# ----------------------------------------------------------------------------------

# write a pytorch state_dict as a string
# with full demical precision
# fmtTensor is called internally
def To_Str_State_Dict(state_dict):
    def dfs(A):
        if type(A) in [dict, OrderedDict]:
            return type(A)([[k,dfs(v)] for k,v in A.items()])
        
        elif type(A) in [list, tuple]:
            return type(A)([dfs(v) for v in A])
        
        elif type(A) in [float, int, str, bool]:
            return type(A)(A)
        
        elif A is None:
            return None
        
        elif hasattr(A,'shape'):
            return fmtTensor(A,tensor_keyword=True)
        
        else:
            raise Exception(f'Unrecognized (type = {type(A)}) (state_dict={state_dict})')
    
    return fmt_D(dfs(state_dict))

# ----------------------------------------------------------------------------------
#                  Write string into zip file
# ----------------------------------------------------------------------------------

# write a string "x" into a zipped file {fname}.zip
def zip_str_write(fname, x, check_path = True):
    local_fname  =  basename(fname)
    target_zip   =  fname + '.zip'
    
    assert type(x) == str
    if check_path:
        assert not isfile(fname),fname
        assert not isfile(target_zip),target_zip

    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(local_fname, x)

# ----------------------------------------------------------------------------------
#                  Read string from zipped file
# ----------------------------------------------------------------------------------

# read text in a zipped file
#    the following convention is assumed:
#        zipped_filename = original_filename + '.zip'
def reader_zipped_text(fname):
    fname       =  fname.removesuffix('.zip')
    saved_zip   =  fname + '.zip'
    local_name  =  basename(fname)
    assert isfile(saved_zip) and not isfile(fname)
    return zipfile.Path(saved_zip, at = local_name).read_text().split('\n')

# ----------------------------------------------------------------------------------
#                  Convert entire object into a dictionary with attributes
# ----------------------------------------------------------------------------------

# return a dictionary with all relevant attributes
# from a nested tree of objects
# eval(Dictify_obj.get(obj))
class Dictify_obj:
    @staticmethod
    def get(obj):
        # eval(str(Dictify_obj.dfs(obj)))
        result = str(Dictify_obj.dfs(obj))
        result = result.replace("<class 'dict'>", 'dict') # defaultdict can create this string sometimes
        return result
    
    @staticmethod
    def dfs(m):
        if hasattr(m,'__sympy__'):
            result = m.__repr__()
        
        elif Dictify_obj.is_obj(m):
            result = str(m)
        
        elif hasattr(m,'shape'):
            if hasattr(m,'A'):  m = m.A
            result = m.tolist()

        elif (type(m) in [OrderedDict, dict]) or hasattr(m,'__dict__'):
            if not (type(m) in [OrderedDict, dict]): m = m.__dict__
            result = dict() if type(m)!=OrderedDict else OrderedDict()
            for key,v in m.items():
                result[key]  =  Dictify_obj.dfs(v)

        elif type(m) in [list,tuple]:
            result = type(m)(lmap(Dictify_obj.dfs, m))
            
        else:
            result  =  m 
        return result
    
    @staticmethod
    def is_obj(m):
        s       =  str(m)
        if not s: return False
        banned  =  ['<module'                        ,
                    '<built-in method'               ,
                    '<bound method'                  ,
                    '<function'                      ,
                    "<class 'memoryview'>"           ,
                    "defaultdict(<class 'dict'>, {})"]
        return (s[0]=='<') and (s[-1]=='>') and any((s[:len(b)]==b) for b in banned)