



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import os
import pandas as pd
import pingouin as pg






########################
######## STATS ######## 
########################



def which_pre_test(df, dv, grouping):

    df = df.reset_index()

    normalities = pg.normality(data = df , dv = dv, group = grouping)['normal']
    
    if sum(normalities) == normalities.size:
        normality = True
    else:
        normality = False
        
    homoscedasticity = pg.homoscedasticity(data = df, dv = dv, group = grouping)['equal_var'].values[0]
    
    if normality and homoscedasticity:
        test_to_use = 'anova'
    else:
        test_to_use = 'friedman'

    return normality, test_to_use





def pre_and_post_hoc(df, within, seuil, sujet):
    
    p_values = {}
    rows_anov = []
    ttests = []
    
    for metric in df.columns:
        
        normality, test_to_use = which_pre_test(df=df, dv = metric , grouping=within)
        
        if test_to_use == 'anova':
            rm_anova = pg.rm_anova(data=df.reset_index(), dv = metric, within = within, subject = sujet)
            p_values[metric] = rm_anova.loc[:,'p-unc'].round(3).values[0]
            test_type = 'rm_anova'
            effsize = rm_anova.loc[:,'np2'].round(3).values[0]
        elif test_to_use == 'friedman':
            friedman = pg.friedman(data=df.reset_index(), dv = metric, within = within, subject = sujet)
            p_values[metric] = friedman.loc[:,'p-unc'].round(3).values[0]
            test_type = 'friedman'
            effsize = np.nan
            
        if p_values[metric] <= seuil : 
            significativity = 1
        else:
            significativity = 0
               
        row_anov = [metric , test_type , p_values[metric] , significativity, effsize]
        rows_anov.append(row_anov)
        
        ttest_metric = pg.pairwise_ttests(data=df.reset_index(), dv=metric, within=within, subject=sujet, parametric = normality, return_desc=True)
        ttest_metric.insert(0, 'metric', metric)
        ttests.append(ttest_metric)
        
    post_hocs = pd.concat(ttests)
    
    colnames = ['metric','test_type','pval', 'signif', 'effsize']
    df_pre = pd.DataFrame(rows_anov, columns = colnames)   

    return df_pre, post_hocs


def test_raw_to_signif(df_pre, post_hocs, seuil):
    mask = df_pre['signif'] == 1
    pre_signif = df_pre[mask]

    post_hocs_signif = post_hocs[post_hocs['p-unc'] < seuil]

    return pre_signif, post_hocs_signif



def post_hoc_interpretation(post_hocs_signif):
    

    conclusions = []
    
    for line in range(post_hocs_signif.shape[0]):
        
        metric = post_hocs_signif.reset_index().loc[line,'metric']
        cond1 = post_hocs_signif.reset_index().loc[line,'A']
        cond2 = post_hocs_signif.reset_index().loc[line,'B']
        
        hedge = np.abs(post_hocs_signif.reset_index().loc[line,'hedges'])

        if hedge <= 0.2:
            intensite = 'faible'
        elif hedge <= 0.8 and hedge >= 0.2:
            intensite = 'moyen'
        elif hedge >= 0.8:
            intensite = 'fort' 
        
        meanA = post_hocs_signif.reset_index().loc[line,'mean(A)']
        meanB = post_hocs_signif.reset_index().loc[line,'mean(B)']
            
        if meanA > meanB:
            comparateur = 'supérieur(e)'
        elif meanA < meanB:
            comparateur = 'inférieur(e)'

        conclusions.append(f"{metric} mesuré(e) en {cond1} est {comparateur} à {metric} mesuré(e) en {cond2} (effet {intensite})")
            
    return conclusions

#df=df_res
def smart_stats(df, within, seuil, sujet):
    
    df_pre, df_post = pre_and_post_hoc(df, within, seuil, sujet)
    pre_signif, post_signif = test_raw_to_signif(df_pre, df_post, seuil)
        
    if post_signif.shape[0] == 0:
        conclusions = None
    else:
        conclusions = post_hoc_interpretation(post_signif)

    return df_pre, pre_signif, df_post, post_signif, conclusions





