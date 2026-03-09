# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------
from  os.path  import  dirname, abspath, isfile, isdir
from  os.path  import  join                        as  pjoin
from  os       import  listdir                     as  os_listdir

# ----------------------------------------------------------------------------------
#                  Basic Definitions
# ----------------------------------------------------------------------------------

lmap              =  lambda f,x: list(map   (f,x))
lfilter           =  lambda f,x: list(filter(f,x))

listdir_full          =  lambda   x:  sorted(pjoin(x,y) for y in os_listdir(x))
listdir_full_folders  =  lambda   x:  lfilter(isdir , listdir_full(x))
listdir_full_files    =  lambda   x:  lfilter(isfile, listdir_full(x))


# ------------------------------------------------------------------------
#                  Cleaner
# ------------------------------------------------------------------------

def run_cleaner(root_folder, A, also_python_code):
    all_fname = sorted(filter(lambda x: x.endswith('.dat.zip'), listdir_full_files(root_folder)),
                       key = lambda x: int(x.split('_state_iter_')[1].split('.')[0])            )

    if also_python_code:
        python_code = list(filter(lambda x: x.endswith('python_code.zip'), listdir_full_files(root_folder)))
        if python_code:
            A.append(python_code)
    
    if all_fname:
        A.extend(all_fname[1:-1])
    else:
        for folder in listdir_full_folders(root_folder):
            run_cleaner(folder, A, also_python_code)

# ------------------------------------------------------------------------
#                  Main Runner
# ------------------------------------------------------------------------

def main_runner(also_python_code):
    field_inv_folder  =  pjoin(dirname(abspath(__file__)), 'field_inversion')
    cache             =  []
    run_cleaner(field_inv_folder, cache, also_python_code)

    cache_fname = pjoin(dirname(abspath(__file__)), 'redundant_fi_saved_files.dat')

    with open(cache_fname, 'w') as f:
        for x in cache:
            f.write(str(x).rstrip('\n') + '\n')
    
    print(f'Done! ({len(cache) = })')

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    also_python_code = False
    main_runner(also_python_code)