import statsmodels.api as sm 
import statsmodels
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def GAM_pt(pse_t, expr, smooth = 'BSplines', df = 5, degree = 3, family = sm.families.NegativeBinomial()):
    """\
    Fit a Generalized Additive Model with the exog to be the pseudo-time. The likelihood ratio test is performed 
    to test the significance of pseudo-time in affecting gene expression value

    Parameters
    ----------
    pse_t
        pseudo-time
    expr
        expression value
    smooth
        choose between BSplines and CyclicCubicSplines
    df
        number of basis function, or degree of freedom
    degree
        degree of the spline function
    family
        distribution family to choose, default is negative binomial.

    Returns
    -------
    y_full
        predict regressed value with full model
    y_reduced
        predict regressed value from null hypothesis
    lr_pvalue
        p-value
    """ 
    from statsmodels.gam.api import GLMGam, BSplines, CyclicCubicSplines

    if smooth == 'BSplines':
        spline = BSplines(pse_t, df = [df], degree = [degree])
    elif smooth == 'CyclicCubicSplines':
        spline = CyclicCubicSplines(pse_t, df = [df])

    exog, endog = sm.add_constant(pse_t),expr
    # calculate full model
    model_full = sm.GLMGam(endog = endog, exog = exog, smoother = spline, family = family)
    try:
        res_full = model_full.fit()
    except:
        # print("The gene expression is mostly zero")
        return None, None, None
    else:
        # default is exog
        y_full = res_full.predict()
        # reduced model
        y_reduced = res_full.null

        # number of samples - number of paras (res_full.df_resid)
        df_full_residual = expr.shape[0] - df
        df_reduced_residual = expr.shape[0] - 1

        # likelihood of full model
        llf_full = res_full.llf
        # likelihood of reduced(null) model
        llf_reduced = res_full.llnull

        lrdf = (df_reduced_residual - df_full_residual)
        lrstat = -2*(llf_reduced - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
        return y_full, y_reduced, lr_pvalue
    

def de_single(setting):
    y = setting["y"]
    distri = setting["distri"]
    verbose = setting["verbose"]
    feature = setting["feature"]
    
    pse_t = np.arange(1, y.shape[0]+1)[:,None]
    if distri == "neg-binomial":
        y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 4, degree = 3, family=sm.families.NegativeBinomial())
    
    elif distri == "log-normal":                
        y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian(link = sm.families.links.log()))
    
    elif distri == "normal":
        try:
            y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian())
        except:
            p_val = None
    elif distri == "binomial":
        y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 3, degree = 2, family=sm.families.Binomial())
        
    else:
        raise ValueError("`distri' is wrong")

    if p_val is not None:
        if verbose:
            print("feature: ", feature, ", pvalue = ", p_val)
        # if p_val <= p_val_t:
        return {"feature": feature, "regression": y_pred, "null": y_null,"p_val": p_val}

def de_analy_para(X, pseudo_order, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True, n_jobs = 1):
    """\
    Conduct differentially expressed gene analysis.

    Parameters
    ----------
    X
        the gene expression data, of the shape (ncells, ngenes)
    pseudo_order
        the data frame that stores the pseudotime of each trajectories.
    p_val_t
        the threshold of p-value
    verbose
        output the differentially expressed gene
    distri
        distribution of gene expression: either "neg-binomial" or "log-normal"
    fdr_correct
        conduct fdr correction for multiple tests or not

    Returns
    -------
    de_genes
        dictionary that store the differentially expressed genes
    """ 
    diff_feats = {}
    for traj_i in pseudo_order.columns:
        diff_feats[traj_i] = []
        sorted_pt = pseudo_order[traj_i].dropna(axis = 0).sort_values()
        # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        ordering = sorted_pt.index.values.squeeze()

        # adata = cellpath_obj.adata[ordering,:]
        # # filter out genes that are expressed in a small proportion of cells 
        # sc.pp.filter_genes(adata, min_cells = int(0.05 * ordering.shape[0]))
        # # spliced stores the count before log transform, but after library size normalization. 

        X_traj = X.loc[ordering, :]

        pool = Pool(n_jobs)
        settings = []
        for idx, feature in enumerate(X_traj.columns.values):
            settings.append({
                "y": np.squeeze(X_traj.iloc[:,idx].values),
                "distri": distri,
                "verbose": verbose,
                "feature": feature
            })
        diff_feats[traj_i] = pool.map(de_single, [x for x in settings])
        diff_feats[traj_i] = [x for x in diff_feats[traj_i] if x is not None]
      
        # sort according to the p_val
        diff_feats[traj_i] = sorted(diff_feats[traj_i], key=lambda x: x["p_val"],reverse=False)

        # fdr correction for multiple tests
        if fdr_correct:
            pvals = [x["p_val"] for x in diff_feats[traj_i]]
            is_de, pvals = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=p_val_t, method='indep', is_sorted=True)
            
            # update p-value
            for gene_idx in range(len(diff_feats[traj_i])):
                diff_feats[traj_i][gene_idx]["p_val"] = pvals[gene_idx]
            
            # remove the non-de genes
            diff_feats[traj_i] = [x for i,x in enumerate(diff_feats[traj_i]) if is_de[i] == True]

    return diff_feats



def de_plot(X, pseudo_order, de_feats, figsize = (20,40), n_feats = 20):
    """\
    Plot differentially expressed gene.

    Parameters
    ----------
    X
        the gene expression data, of the shape (ncells, ngenes)
    pseudo_order
        the data frame that stores the pseudotime of each trajectories
    de_feats
        dictionary that store the differentially expressed genes
    figsize
        figure size
    n_feats
        the number of genes to keep, from smallest p-values
    save_path
        the saving directory 
    """ 
    # # turn off interactive mode for matplotlib
    # plt.ioff()

    ncols = 2
    nrows = np.ceil(n_feats/2).astype('int32')

    figs = []
    for traj_i in de_feats.keys():
        # ordering of genes
        sorted_pt = pseudo_order[traj_i].dropna(axis = 0).sort_values()
        # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        ordering = sorted_pt.index.values.squeeze()
        X_traj = X.loc[ordering, :]

        # make plot
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
        colormap = plt.cm.get_cmap('tab20b', n_feats)
        for idx, feat in enumerate(de_feats[traj_i][:n_feats]):
            # plot log transformed version
            y = np.squeeze(X_traj.loc[:,feat["feature"]].values)
            y_null = feat['null']
            y_pred = feat['regression']

            axs[idx%nrows, idx//nrows].scatter(np.arange(y.shape[0]), y, color = colormap(1), alpha = 0.5)
            axs[idx%nrows, idx//nrows].plot(np.arange(y.shape[0]), y_pred, color = "black", alpha = 1, linewidth = 4)
            axs[idx%nrows, idx//nrows].plot(np.arange(y.shape[0]), y_null, color = "red", alpha = 1, linewidth = 4)
            axs[idx%nrows, idx//nrows].set_title(feat["feature"])                
        
        plt.tight_layout()
        figs.append(fig)
    
    return figs


'''
def de_analy(X, pseudo_order, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True):
    """\
    Conduct differentially expressed gene analysis.

    Parameters
    ----------
    X
        the gene expression data, of the shape (ncells, ngenes)
    pseudo_order
        the data frame that stores the pseudotime of each trajectories.
    p_val_t
        the threshold of p-value
    verbose
        output the differentially expressed gene
    distri
        distribution of gene expression: either "neg-binomial" or "log-normal"
    fdr_correct
        conduct fdr correction for multiple tests or not

    Returns
    -------
    de_genes
        dictionary that store the differentially expressed genes
    """ 

    diff_feats = {}
    for traj_i in pseudo_order.columns:
        diff_feats[traj_i] = []
        sorted_pt = pseudo_order[traj_i].dropna(axis = 0).sort_values()
        # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        ordering = sorted_pt.index.values.squeeze()

        # adata = cellpath_obj.adata[ordering,:]
        # # filter out genes that are expressed in a small proportion of cells 
        # sc.pp.filter_genes(adata, min_cells = int(0.05 * ordering.shape[0]))
        # # spliced stores the count before log transform, but after library size normalization. 

        X_traj = X.loc[ordering, :]

        # loop through all features
        for idx, feature in enumerate(X_traj.columns.values):
            y = np.squeeze(X_traj.iloc[:,idx].values)
            pse_t = np.arange(1, y.shape[0]+1)[:,None]
            if distri == "neg-binomial":
                y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 4, degree = 3, family=sm.families.NegativeBinomial())
            
            elif distri == "log-normal":                
                y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian(link = sm.families.links.log()))
            
            elif distri == "normal":
                try:
                    y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian())
                except:
                    p_val = None
            elif distri == "binomial":
                y_pred, y_null, p_val = GAM_pt(pse_t, y, smooth='BSplines', df = 3, degree = 2, family=sm.families.Binomial())
                
            else:
                raise ValueError("`distri' is wrong")

            if p_val is not None:
                if verbose:
                    print("feature: ", feature, ", pvalue = ", p_val)
                # if p_val <= p_val_t:
                diff_feats[traj_i].append({"feature": feature, "regression": y_pred, "null": y_null,"p_val": p_val})
        
        # sort according to the p_val
        diff_feats[traj_i] = sorted(diff_feats[traj_i], key=lambda x: x["p_val"],reverse=False)

        # fdr correction for multiple tests
        if fdr_correct:
            pvals = [x["p_val"] for x in diff_feats[traj_i]]
            is_de, pvals = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=p_val_t, method='indep', is_sorted=True)
            
            # update p-value
            for gene_idx in range(len(diff_feats[traj_i])):
                diff_feats[traj_i][gene_idx]["p_val"] = pvals[gene_idx]
            
            # remove the non-de genes
            diff_feats[traj_i] = [x for i,x in enumerate(diff_feats[traj_i]) if is_de[i] == True]

    return diff_feats

'''