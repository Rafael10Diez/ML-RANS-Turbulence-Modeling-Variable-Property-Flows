# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  torch

# ------------------------------------------------------------------------
#                  Basic Definitions
# ------------------------------------------------------------------------

nn    =  torch.nn

# ------------------------------------------------------------------------
#                  Seeder Function
# ------------------------------------------------------------------------

def apply_seed_torch(SEED):
    torch.     manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ------------------------------------------------------------------------
#                  Multi-layer neural network
# ------------------------------------------------------------------------

class Muti_Activation(nn.Module):
    def __init__(self, C_in, C_out, layer):
        super().__init__()

        assert len(layer) == C_out

        D_tag = {'T' : nn.Tanh    ,
                 'S' : nn.Sigmoid ,
                 'R' : nn.ReLU    ,
                 'P' : nn.PReLU   }
        
        self.linear      = nn.Linear(C_in, C_out)
        self.activations = nn.ModuleList([D_tag[tag]() for tag in layer])

    def forward(self, x):
        x       =  self.linear(x)
        result  =  []
        for i,a in enumerate(self.activations):
            result.append(a(x[... , i]))
        return torch.stack(result,dim=-1)

# ------------------------------------------------------------------------
#                  Deep Learning Model
# ------------------------------------------------------------------------

class Neural_Network(nn.Module):
    def __init__(self, args):
        # args -> seed, n_features, param_groups, all_layers
        super().__init__()
        apply_seed_torch(args['seed'])
        
        self.A  =  []

        all_C   =  [args['n_features'], args['param_groups']]
        self.A.append(nn.Linear(all_C[-2], all_C[-1], bias=False)) # bias is redundant here

        for layer in args['all_layers']:
            all_C.append(len(layer))
            self.A.append(Muti_Activation(all_C[-2],
                                          all_C[-1],
                                          layer    ))
        
        self.A     =  nn.Sequential(*self.A)

        self.get_weights_loglayer = lambda: [self.A[0].weight]
    
    def forward(self, x):
        x = torch.log10(x.abs()+1e-8)
        return self.A(x)

