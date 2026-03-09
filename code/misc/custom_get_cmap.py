# ----------------------------------------------------------------------------------
#                  Generic Libraries
# ----------------------------------------------------------------------------------

from    sys          import  path             as  sys_path
from    matplotlib   import  colors, cm
from    os.path      import  dirname, abspath
from    os.path      import  join             as  pjoin
import  numpy        as      np

# ----------------------------------------------------------------------------------
#                  Custom Libraries
# ----------------------------------------------------------------------------------

sys_path.append(dirname(abspath(__file__)))
from utils import reader, deepdirname
sys_path.pop()

# ----------------------------------------------------------------------------------
#                  Basic Definitions
# ----------------------------------------------------------------------------------

paraview_colormaps  =  eval(' '.join(reader(pjoin(deepdirname(abspath(__file__),3), 'data', 'input', 'paraview_colormaps.dat'))))

# ------------------------------------------------------------------------
#                  Color Maps
# ------------------------------------------------------------------------

def custom_get_cmap(cmap_name, MODE_PLOT, black = None):
    if 'plt_' == cmap_name[:4]:
        scale = cm.get_cmap(cmap_name.removeprefix('plt_'))(np.linspace(0,1,1000))[:,:3].tolist()
    else:
        assert 'pv_' == cmap_name[:3]
        scale = paraview_colormaps[cmap_name]
    
    if black:
        assert black <= 1
        scale = scale + (np.array(scale)*(1-black)).tolist()
    
    if    MODE_PLOT == 'plt':
        return colors.ListedColormap(scale, name = cmap_name)
    elif  MODE_PLOT == 'mlab':
        cempty  =  255
        return np.array([ list(np.array(row)*cempty)+[cempty] for row in scale])
    else:
        raise Exception(f'ERROR: Unrecognized (MODE_PLOT = {MODE_PLOT})')