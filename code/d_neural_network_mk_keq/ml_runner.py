# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  torch
import  socket
from    sys          import  path                                       as  sys_path
from    os.path      import  join                                       as  pjoin
from    os           import  mkdir                                      as  os_mkdir
from    os.path      import  basename, abspath, dirname, isdir
from    time         import  time
from    copy         import  deepcopy
from    collections  import  OrderedDict

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

sys_path.append(dirname(abspath(__file__)))
from  deep_learning  import  Neural_Network
from  data_loaders   import  All_Data_Loaders
from  read_fi_data   import  Read_FI_Data
sys_path.pop()

sys_path.append(dirname(dirname(abspath(__file__))))
from  misc.utils import ( lmap              ,
                          improved_pformat  ,
                          fmtTensor         ,
                          To_Str_State_Dict ,
                          zip_str_write     ,
                          FPrint            ,
                          Dictify_obj       ,
                          format_dt         )
sys_path.pop()

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

own_socket       =  socket.gethostname()
calc_net_params  =  lambda net: sum(p.numel() for p in net.parameters() if p.requires_grad)

fmt_D            =  lambda   D:  str(D).replace(' ','')
fmt_p            =  lambda   x:  f'{(x*100):.5f} %'

# ------------------------------------------------------------------------
#                  ML Runner
# ------------------------------------------------------------------------

class ML_Runner:
    def __init__(self, args):
        
        self.args           =  args
        self.output_folder  =  args['output_folder']
        
        if not isdir(self.output_folder):
            os_mkdir(self.output_folder)
            os_mkdir(pjoin(self.output_folder,'saved'))
        
        # Register Arguments:
        self.fprint  =  fprint  =  FPrint(pjoin(self.output_folder                   ,
                                                basename(self.output_folder) + '.log'))
        self.device             =  args['device']
        
        assert 'cuda' in self.device
        
        # Header (first part)
        fprint( '\n--------------------------------------------------\n    ML Runner\n--------------------------------------------------\n')
        fprint(f'\nLog Filename (this file)   :  {fprint.fname}')
        fprint(f'\nHost                       :  {own_socket  }')
        fprint(f'\nDevice                     :  {self.device }')
        fprint(   'Input Arguments (ML_Runner):')
        fprint('# Begin_Input_Arguments')
        fprint(improved_pformat(args))
        fprint('# End_Input_Arguments'  )

        # build internal components
        self.loaders  =  All_Data_Loaders(self.__mk_args_loaders())

        self.__n_calls_mk_new  =  0
        self.mk_new_net_opt()
        
        # Define criterion
        self.criterion  =  eval(args['loss_function'])

        # make data loaders
        self.total_params  =  calc_net_params(self.net)

        # Header (second part)
        fprint(f'\nNumber of Parameters: {self.total_params}')
        
        fprint.only_to_file('')
        fprint.only_to_file('\n--------------------------------------------------\n    Details Neural Network\n--------------------------------------------------\n')
        fprint.only_to_file( self.net )
        fprint.only_to_file('')
        
        # write initial data
        zip_str_write(pjoin(self.output_folder, 'saved', 'initial_data.dat'),
                      Dictify_obj.get(self)                                 )
        
        self.pred_stats = {}
        self.runner()

    def mk_new_net_opt(self, all_state_dict = None):
        args  =  self.args

        # logic: when a state_dict is restored, its original seed should also be reloaded
        self.__seed_id          =  self.__n_calls_mk_new if (all_state_dict is None) else int(all_state_dict['seed_id'])
        self.__n_calls_mk_new  +=  1

        args_net               =  deepcopy(args['args_net'])
        args_net['n_features'] =  self.loaders.n_features
        args_net['seed']       =  self.__seed_id
        net                    =  Neural_Network(args_net)

        if all_state_dict:
            net.load_state_dict(eval(all_state_dict['net']))
        
        net  =  net.to(self.device)
        
        p_change    =  [p for p in net.parameters() if p.requires_grad]
        p_loglayer  =  net.get_weights_loglayer()
        
        if args['optim_w_type'] == 'adam':
            optim_w  =  torch.optim.Adam( p_change                                ,
                                          lr            =  args['lr_w']           ,
                                          betas         =  args['betas_w']        ,
                                          weight_decay  =  args['weight_decay_w'] ,
                                          eps           =  args['eps']            )
            if all_state_dict:
                optim_w.load_state_dict(eval(all_state_dict['optim_w']))
        else:
            raise Exception(f"Error: Unrecognized Optimizer (optim_w_type = {args['optim_w_type']})")
        
        self.net         =  net
        self.p_change    =  p_change
        self.p_loglayer  =  p_loglayer
        self.optim_w     =  optim_w

    def runner(self):
        line             =  '--------------------'
        
        min_saver_dt     =  self.args['min_saver_dt' ]
        epoch_predict    =  self.args['epoch_predict']
        epoch_report     =  self.args['epoch_report' ]
        
        best_seed                      = {'track_epochs'  :  set(lmap(lambda i: i*self.args['seeder']['epochs_per_seed'],
                                                                      range(1,    self.args['seeder']['n_seeds_try']+1))),
                                          'record'        :  float('inf')                                                 ,
                                          'all_state_dict':  None                                                         }
        best_seed['epoch_final_load']  =  max(best_seed['track_epochs']) + 1
        
        assert not self.args['epochs_per_seed']%self.args['epoch_save_freq']

        saved_epochs      = list(range(0,self.args['Epochs']+1,self.args['epoch_save_freq']))[1:]
        while saved_epochs[-1] > self.args['Epochs']:
            saved_epochs.pop()
        
        if type(self.args['epoch_n_saves']) in [str,int]:
            saved_epochs = saved_epochs[-min(len(saved_epochs)              ,
                                             int(self.args['epoch_n_saves'])):]
        
        print(f'INFORMATION: saved_epochs = {saved_epochs}')
        
        t0 = t1 = t2 = time()
        prev_epoch   =  0
        
        for epoch in range(1,self.args['Epochs']+1):  # loop over the dataset multiple times
            
            if epoch > saved_epochs[-1]: break # further epochs will not be saved
            
            if epoch == best_seed['epoch_final_load']:
                assert best_seed['all_state_dict']
                self.mk_new_net_opt(all_state_dict = best_seed['all_state_dict'])
                self.fprint(f'{line} Reloading Best (Seed = {self.__seed_id}) {line}')
            
            self.net.train()
            
            self.__current_epoch  =  epoch
            
            # main step training weights
            
            self.optim_w.zero_grad()
            running_loss = self.get_loss('train', backward=True)
            self.optim_w.step()
            
            # report
            if (epoch % epoch_report) == 0:
                self.fprint(f'Epoch : {epoch:8d} (loss: {running_loss:.4e}) {self.mag_report()} (total time: {format_dt(time()-t0)}) (avg. time/epoch: {((time()-t1)/(epoch-prev_epoch)):10.6f})')
                prev_epoch  =  float(epoch)
                t1 = time()
            
            # predict
            if (epoch % epoch_predict) == 0:
                assert (epoch % epoch_report ) == 0
                self.global_predict(pred_stats=True)
            
            # saving
            if epoch in saved_epochs:
                assert (time()-t2) >= min_saver_dt
                assert (epoch % epoch_report ) == 0
                assert (epoch % epoch_predict) == 0
                self.saver()
                t2 = time()
            
            # best seed tracker
            if epoch in best_seed['track_epochs']:
                self.fprint(f'{line} Evaluating Results Seed {line}')
                current_metric  =  self.score_seed_performance()
                if current_metric < best_seed['record']:
                    self.fprint(f'    New Record Seed: (new best = {fmt_p(current_metric)}) (old = {fmt_p(best_seed["record"])})')
                    best_seed['record']           =  current_metric
                    best_seed['all_state_dict']  =  {'net'    :  To_Str_State_Dict(self.net    .state_dict()),
                                                     'optim_w':  To_Str_State_Dict(self.optim_w.state_dict()),
                                                     'seed_id':  int(self.__seed_id)                         }
                else:
                    self.fprint(f'    Seed Discarded: (metric = {fmt_p(current_metric)}) (best = {fmt_p(best_seed["record"])})')
                self.fprint(f'{line} Starting New Seed {line}')
                self.mk_new_net_opt()
        
        self.fprint(f'# Finished_Training (elapsed time = {format_dt(time()-t0)})')
        self.fprint.only_to_file('')
        self.fprint.only_to_file('# Begin_Pred_Stats')
        self.fprint.only_to_file(improved_pformat(self.pred_stats))
        self.fprint.only_to_file('# End_Pred_Stats')
        zip_str_write(pjoin(self.output_folder, 'saved', 'end_data.dat'),
                      Dictify_obj.get(self)                             )
    
    def score_seed_performance(self):
        n_use        =  self.args['seeder']['epochs_avg']//self.args['epoch_predict']
        metric_name  =  self.args['seeder']['metric_name']
        use_dsets    =  self.args['seeder']['use_dsets']
        assert list(use_dsets) == ['train']
        all_val      =  []
        for dset in use_dsets:
            all_val  += [abs(val) for val in lmap(lambda d: d[metric_name]      ,
                                                  self.pred_stats[dset][-n_use:])]
        return sum(all_val)/len(all_val)

    def get_loss(self, dset, backward = False):
        assert  (dset  ==  'train')  and  backward
        data          =  self.loaders.get_loader[dset]
        data          =  [[data.X_stack, data.Y_stack]]
        # data is already on the device
        running_loss  =  0.
        for inputs, labels in data:
            loss            =  self.criterion(self.net(inputs) - labels, dict(p_change   = self.p_change,
                                                                              p_loglayer = self.p_loglayer))
            if backward:
                loss.backward()
            running_loss  +=  float(loss.item())
        return running_loss
    
    def mag_report(self):
        result = []
        for     tag, p_array in [['main', self.p_change  ],
                                 ['log' , self.p_loglayer]]:
            s = f'(w{tag}: '
            t = []
            for mode in ['L2','L1']:
                t.append(f"{mode} = {self.get_wd_mag(p_array,mode):.4e}")
            s +=  ', '.join(t) + ')'
            result.append(s)
        return ' '.join(result)
           
    @staticmethod
    def get_wd_mag(p_array, mode):
        f = {'L1': lambda x:  x.abs(),
             'L2': lambda x:  x**2   }[mode]
        return sum(float(f(p).sum().item()) for p in p_array if p.requires_grad)
    
    def global_predict(self, fp = None, pred_stats = None):
        assert type(pred_stats) == bool
        fp = fp or self.fprint
        self.predict('train', fp, pred_stats)
        self.predict('valid', fp, pred_stats)
        self.predict('test' , fp, pred_stats)
    
    def predict(self, dset, fp, pred_stats):
        if not dset in self.loaders.get_loader:
            return
        
        t0    =  time()
        data  =  self.loaders.get_loader[dset]
        if data is None: return
        data  =  [[data.X_stack, data.Y_stack]]
        
        abstotal, absdiff  =  0., 0.
        
        self.net.eval()
        
        for inputs, labels in data:
            
            delta      =  self.net(inputs) - labels
            
            abstotal  +=  float(labels.abs().sum().item())
            absdiff   +=  float(delta .abs().sum().item())
        
        self.net.train() 
        
        new_stats = {'|diff|/|total|':  absdiff / max(1e-12,abstotal)  ,
                     'epoch'         :  self.__current_epoch}
        
        if pred_stats:
            if not dset in self.pred_stats: self.pred_stats[dset] = []
            self.pred_stats[dset].append(new_stats)
        
        mk_str = lambda : ' '.join(f'({k}: {(v*100):10.5f} %)' for k,v in new_stats.items() if k!='epoch')
        fp(f"Prediction (dset = {dset:5s}): {mk_str()} (elapsed time = {(time()-t0):10.6f})")
    
    def saver(self):
        
        outpath       =  lambda suffix:  pjoin( self.output_folder                                               ,
                                               'saved'                                                           ,
                                               f'seed_{self.__seed_id}_epoch_{self.__current_epoch}_{suffix}.dat')
        
        fname_net     =  outpath('statedict_net'    )
        fname_optim   =  outpath('statedict_optim_w')
        fname_fields  =  outpath('predicted'        )
        
        self.fprint.only_to_file('\n------------------------- Begin Saver -------------------------')
        self.global_predict(fp = self.fprint.only_to_file, pred_stats=False)
        self.fprint.only_to_file( f'---->  Net_Dict       :  {fname_net}'   )
        self.fprint.only_to_file( f'---->  Optim_Dict     :  {fname_optim}' )
        self.fprint.only_to_file( f'---->  Field Vars     :  {fname_fields}' )
        self.fprint.only_to_file('------------------------- End   Saver -------------------------')
        self.fprint.only_to_file('')
        
        zip_str_write(fname_net                                   , # fname_net
                      To_Str_State_Dict(self.net.state_dict())    ) # sd_net
        
        zip_str_write(fname_optim                                 , # fname_optim
                      To_Str_State_Dict(self.optim_w.state_dict())) # sd_optim_w
        
        self.net.eval()
        D = {}
        for dset in ['train','valid','test']:
            if dset in self.loaders.get_loader:
                inputs  = self.loaders.get_loader[dset].X_stack
                D[dset]                           = {}
                D[dset]['predictions_real_scale'] = fmtTensor(self.net(inputs)     ,
                                                              tensor_keyword=False )
        
        D = fmt_D(D)
        self.net.train()
        
        zip_str_write(fname_fields, # fname_fields
                      D           ) # D
    
    def __mk_args_loaders(self):
        order = [ 'Y_star'     ,  'prod_k/Sk' , 
                  'u/Su'       ,  'dest_k/Sk' ,
                  'k/Mk'       ,  'diff_k/Sk' ,
                  'e/Me'       ,  'Ret_star'  ,
                  'r/r_w'      ,  'Su'        ,
                  'mu/mu_w'    ,  'Sk'        ,
                  'mu_t/mu_w'  ,  'Mk'        ] 
        
        return dict( device     =  self.device                               ,
                     dtype      =  torch.float                               ,
                     order      =  order                                     ,
                     get_all_D  =  Read_FI_Data.get(self.args['used_cases']) )

