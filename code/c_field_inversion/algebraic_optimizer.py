# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    copy     import  deepcopy
import  sympy    as      sp
from    string   import  ascii_letters
from    re       import  finditer as re_finditer
# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

power  =  lambda a,b: a**b

# ------------------------------------------------------------------------
#                  Wrapper for Algebraic Optimizer
# ------------------------------------------------------------------------

class Optimized_Algebraic_Function:
    def __init__(self, args):
        self.args = args
        # Dopt is the results of the optimized expression
        # Dopt.keys -> [optimized_original, tempvars, original_vars]
        self.Dopt  = Internal_Engine_Algebraic_Optimizer.optimize(args['D_terms'], dict(n_nested = args['n_nested'],
                                                                                        n_found  = args['n_found']))
        
        tempvars                =  self.Dopt['tempvars'] # keys are in execution order
        
        self._vars_dof           =  deepcopy(list(self.Dopt['original_vars']))
        self._vars_additional    =  deepcopy(list(tempvars.keys()))
        self._var_functions      =  ['exp','Abs', 'sign', 'sqrt', 'Heaviside', 'Max']

        assert sorted(self._vars_dof) == sorted(args['known_vars'])

        self._D_additionals    =  {}

        for k,v in tempvars.items():
            assert not k in self._vars_dof
            self._D_additionals[k]  =  self.__mk_function(v)
        
        self.__target_function = self.__mk_function(self.Dopt['optimized_original'])

    def __mk_function(self, S):
        S             =  str(S)
        all_vars_tag  =  ([[v, 'D'         ]  for v in  (self._vars_dof+self._var_functions)]  +
                          [[v, 'additional']  for v in  self._vars_additional]                 )
        all_vars_tag.sort(key = lambda x: (-len(x[0]),x))

        all_codes = {}
        for i in range(len(all_vars_tag)):
            var, tag = all_vars_tag[i]
            assert all(not (var in next_pair[0]) for next_pair in all_vars_tag[i+1:])
            new_code            = f'_code{i:08d}code_'
            all_codes[new_code] = f"{tag}['{var}']"
            S  = S.replace(var, new_code)
        
        for code, original in all_codes.items():
            S = S.replace(code, original)

        return eval(f'lambda D, additional: {S}')

    def __call__(self, D):
        additional = {}
        for k,f in self._D_additionals.items(): # order is preserved
            additional[k] = f(D, additional)
        return self.__target_function(D, additional)

# ------------------------------------------------------------------------
#                  Internal Engine for Algebraic Optimizer
# ------------------------------------------------------------------------

class Internal_Engine_Algebraic_Optimizer:
    @staticmethod
    def optimize(D, args):
        n_nested, n_found = args['n_nested'], args['n_found']
        assert type(D) == dict
        assert (n_nested >= 2) and (n_found >= 2)

        encoded  =  Internal_Engine_Algebraic_Optimizer.mk_encoded()

        D         =  Internal_Engine_Algebraic_Optimizer.__as_srepr(deepcopy(D))
        orig_vars = Internal_Engine_Algebraic_Optimizer.get_vars(D)

        A         =  Internal_Engine_Algebraic_Optimizer.__apply([D], 1, encoded, n_nested, n_found)
        
        A         =  Internal_Engine_Algebraic_Optimizer._reorder(A)

        for i in range(len(A)):
            A[i] = Internal_Engine_Algebraic_Optimizer.__as_srepr(A[i], inverse=True)
        
        return dict(optimized_original = A[0]                                      ,
                    tempvars           =  {k:v for d in A[1:] for k,v in d.items()},
                    original_vars      =   orig_vars                               )
    @staticmethod
    def get_vars(D):
        D  =  str(D)

        # general check (for current sympy version)
        assert ('Symbol(' in D) and not ('Symbols(' in D)

        # find symbol start
        all_i = list(map(lambda m: m.start(), re_finditer(r'Symbol', D)))

        assert len(all_i) == D.count('Symbol(')

        all_j = [i for i,c in enumerate(D) if c==')'][::-1]

        result = set()

        for i in all_i:
            while all_j[-1] < i:
                all_j.pop()
            j = all_j.pop()
            result.add(D[i:j].split('(')[1].split(',')[0].replace(' ','').replace('"','').replace("'",''))
        
        return sorted(result)

    @staticmethod
    def mk_encoded():
        charset  =  sorted(set(ascii_letters + ''.join(map(str,range(10)))))
        return [(c1+c2) for c1 in charset for c2 in charset]

    @staticmethod
    def _reorder(A):
        assert type(A) == list
        a0, B        =  A[0], A[1:]
        unknown      =  set(k for d in B for k in d.keys())
        order        =  []
        iter, N_max  =  -1, 1000
        while B:
            iter        += 1
            assert iter <= N_max
            for i in range(len(B)):
                k,v = list(B[i].items())[0]
                if not any((u in v) for u in (unknown-{k,})):
                    unknown.remove(k)
                    order  .append(B[i])
                    B = B[:i] + B[i+1:]
                    break
        return [a0] + order

    @staticmethod
    def __as_srepr(D, inverse = False):
        if type(D) == dict:
            for k in D.keys():
                D[k] = Internal_Engine_Algebraic_Optimizer.__as_srepr(D[k],inverse=inverse)
        else:
            assert not (type(D) in [float, int, tuple, list])
            if not inverse:
                D = sp.srepr(D)
                assert type(D) == str
            else:
                assert type(D) == str
                Add       =  sp.Add    
                Abs       =  sp.Abs    
                Mul       =  sp.Mul    
                Pow       =  sp.Pow    
                Float     =  sp.Float  
                Integer   =  sp.Integer
                Max       =  sp.Max
                Heaviside =  sp.Heaviside
                Rational  =  sp.Rational
                Symbol    =  sp.Symbol 
                exp       =  sp.exp    
                sign      =  sp.sign   
                D         =  eval(D)
        return D
    
    @staticmethod
    def __mk_symbol(tag):
        assert type(tag) == str
        return f"Symbol('{tag}', real = True)"
    
    @staticmethod
    def __ops_count(s):
        assert type(s) == str
        # Integer(-1) appears in redundant operations 
        #     Substract --> Add(??, Mul(-1,??))
        #     Division  --> Pow(??, -1)
        return s.count('(') - s.count('Integer') - s.count('Float') - s.count('Rational') - s.count('Integer(-1)') - s.count('Integer(1)')
    
    @staticmethod
    def __get_keyword(A, i):
        assert type(A) == str
        j  = i
        i -= 1
        while i>=0 and (A[i] in ascii_letters):
            i -= 1
        return A[(i+1):j]

    @staticmethod
    def __get_best_str(A, n_nested, n_found):
        assert type(A) == str

        best_score, best_str  =  (-1,-1), None

        for i,c in enumerate(A):
            if c=='(':
                # find previous keyword
                keyword = Internal_Engine_Algebraic_Optimizer.__get_keyword(A, i)
                assert keyword in ['Add', 'Abs', 'Mul', 'Pow', 'Float', 'Integer', 'Rational', 'Symbol', 'Max', 'Heaviside', 'exp', 'sign']
                
                balance = 1
                j       = i + 1 # start one character forward (because balance = 1)
                i      -= len(keyword) # real beggining of expression
                while balance:
                    if A[j]=='(': balance += 1
                    if A[j]==')': balance -= 1
                    j += 1
                s          = A[i:j]
                n_repeated = A.count(s)

                if n_repeated >= n_found:
                    n_ops = Internal_Engine_Algebraic_Optimizer.__ops_count(s)
                    if n_ops >= n_nested:
                        rank  =  n_ops, n_repeated
                        if rank > best_score:
                            best_score, best_str = rank, s
        return best_str

    @staticmethod
    def __apply(A, new_id, encoded, n_nested, n_found):
        
        A         =  str(A)
        best_str  = Internal_Engine_Algebraic_Optimizer.__get_best_str(A, n_nested, n_found)
        
        if best_str:
            tag     =  f"aux{encoded[new_id]}"
            A       =  A.replace(best_str, Internal_Engine_Algebraic_Optimizer.__mk_symbol(tag))
            A       =  eval(A)
            A.append({tag: best_str})
            new_id += 1
            A       =  Internal_Engine_Algebraic_Optimizer.__clean(A, n_nested, n_found)
            A       =  Internal_Engine_Algebraic_Optimizer.__apply(A, new_id, encoded, n_nested, n_found)
        else:
            A = eval(A)
        return A
    
    @staticmethod
    def __clean(A, n_nested, n_found):
        assert type(A) == list
        count_keytag  =  lambda k: sum(str(val).count(k) for d in A for val in d.values())

        for i in range(1,len(A)):
            assert type(A[i]) == dict
            assert  len(A[i]) == 1
            
            k, v = list(A[i].items())[0]

            assert type(k) == type(v) == str

            # check if expression {k:v} still meets target parameters
            if (Internal_Engine_Algebraic_Optimizer.__ops_count(v) < n_nested) or (count_keytag(k) < n_found):
                # remove all instances of "k"
                for j in range(len(A)):
                    if i!=j:
                        A[j] = eval(str(A[j]).replace(Internal_Engine_Algebraic_Optimizer.__mk_symbol(k), v))
                # remove A[i] itself
                A = A[:i] + A[i+1:]
                # return recursive clean-up
                return Internal_Engine_Algebraic_Optimizer.__clean(A, n_nested, n_found)
        return A
