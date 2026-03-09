import random
random.seed(10)

D_casetypes = dict(supersonic =  [ 'Supersonic_M0.7R400'    ,
                                   'Supersonic_M0.7R600'    ,
                                   'Supersonic_M1.7R200'    ,
                                   'Supersonic_M1.7R400'    ,
                                   'Supersonic_M1.7R600'    ,
                                   'Supersonic_M3.0R200'    ,
                                   'Supersonic_M3.0R400'    ,
                                   'Supersonic_M3.0R600'    ,
                                   'Supersonic_M4.0R200'    ],

                   t31        =  [ 'T31_CP150'              ,
                                   'T31_CP395'              ,
                                   'T31_CP550'              ,
                                   'T31_CReStar_tau'        ,
                                   'T31_Cv'                 ,
                                   'T31_GL'                 ,
                                   'T31_LL1'                ,
                                   'T31_LL2'                ,
                                   'T31_SReStar_tauCv'      ,
                                   'T31_SReStar_tauGL'      ,
                                   'T31_SReStar_tauLL'      ],
                                 
                   t61        =  [ 'T61_CP395_Pr4'          ,
                                   'T61_CReStar_tauCPrStar' ,
                                   'T61_GLCPrStar'          ,
                                   'T61_VLambdaSPrStar_LL'  ],
                   
                   jimenez    =  [ 'Jimenez_Re180'          ,
                                   'Jimenez_Re550'          ,
                                   'Jimenez_Re950'          ,
                                   'Jimenez_Re2000'         ,
                                   'Jimenez_Re4200'         ])

assert sum(map(len, D_casetypes.values())) == 29

n_Kfolds     =  4
n_samples    =  2
data_Kfolds  =  [[] for _ in range(n_Kfolds)]

for A in D_casetypes.values():
    pairs       =  set()
    min_unique  =  min(len(A), n_Kfolds*n_samples)
    extremes = [A[0],A[-1]]
    if 'T31_' in A[0]:
        extremes.append('T31_CReStar_tau')
    while True:
        # build K-fold pairs where every combination is unique
        while len(pairs) < n_Kfolds:
            new  = tuple(map(lambda i: A[i],
                             sorted(random.sample(range(len(A)), n_samples))
                            ))
            if not ((new in pairs)):# or all(c in extremes for c in new)):
                pairs.add(new)
        
        # check if combinations meet requirements
        seen_cases = set(c for new in pairs for c in new)
        if (len(seen_cases) < min_unique) or (not all((c in seen_cases) for c in extremes)):
            pairs = set()
        else:
            break
    for i in range(n_Kfolds):
        data_Kfolds[i].extend(list(pairs.pop()))

len_col  =  [max(map(lambda row: len(row[i]), data_Kfolds)) for i in range(len(data_Kfolds[0]))]
print(len_col)
for row in data_Kfolds:
    quote   = lambda c: f"'{c}'"
    gen_fmt = lambda L: eval('lambda c: f"{quote(c):'+str(L+2)+'}"')
    s = '[' + ', '.join(gen_fmt(len_col[i])(c) for i,c in enumerate(row)) + '],'
    print(s)
# data_Kfolds = [['Supersonic_M1.7R600',
#                 'Supersonic_M4.0R200',
#                 'T31_GL',
#                 'T31_LL2',
#                 'T61_CP395_Pr4',
#                 'T61_VLambdaSPrStar_LL',
#                 'Jimenez_Re180',
#                 'Jimenez_Re550'],
#
#                ['Supersonic_M1.7R200',
#                 'Supersonic_M1.7R400',
#                 'T31_SReStar_tauCv',
#                 'T31_SReStar_tauLL',
#                 'T61_GLCPrStar',
#                 'T61_VLambdaSPrStar_LL',
#                 'Jimenez_Re2000',
#                 'Jimenez_Re950'],
#
#                ['Supersonic_M0.7R600',
#                 'Supersonic_M3.0R600',
#                 'T31_CP150',
#                 'T31_LL1',
#                 'T61_CReStar_tauCPrStar',
#                 'T61_GLCPrStar',
#                 'Jimenez_Re2000',
#                 'Jimenez_Re4200'],
#
#                ['Supersonic_M0.7R400',
#                 'Supersonic_M3.0R200',
#                 'T31_CP395',
#                 'T31_CReStar_tau',
#                 'T61_CReStar_tauCPrStar',
#                 'T61_VLambdaSPrStar_LL',
#                 'Jimenez_Re180',
#                 'Jimenez_Re950']]