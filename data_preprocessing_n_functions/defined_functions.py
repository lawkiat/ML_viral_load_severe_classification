import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, recall_score, precision_score, precision_recall_curve

# Function to retrieve summary of model's performance metrics
def sum_metric(model_list):
    """Extract and aggregate performance metrics across simulated models."""
    rows = []
    for i, result in enumerate(model_list):
        rows.append({
            'run': i + 1,
            'accuracy_train': result['accuracy_training'] * 100,
            'accuracy_test': result['accuracy_testing'] * 100,
            'sensitivity': result['sensitivity'] * 100,
            'specificity': result['specificity'] * 100,
            'precision': result['precision'] * 100,
            'NPV': result['npv'] * 100,
            'AUC_value': result['AUC_value'] * 100,
            'AUPRC_value': result['PRC_val'] * 100
        })
    met_df = pd.DataFrame(rows)
    
    # Calculate summary metrics (Mean, SD, 2.5% quantile, 97.5% quantile)
    summary_dict = {}
    for col in met_df.columns:
        if col == 'run':
            continue
        summary_dict[f"{col}_mean"] = round(met_df[col].mean(), 4)
        summary_dict[f"{col}_sd"] = round(met_df[col].std(ddof = 1), 4)
        summary_dict[f"{col}_low"] = round(met_df[col].quantile(0.025), 4)
        summary_dict[f"{col}_upp"] = round(met_df[col].quantile(0.975), 4)
        
    met_sum = pd.DataFrame([summary_dict])
    return {"metric_df": met_df, "metric_summary": met_sum}

# Function to collate all model's performance metrics
def met_collate_func(met_df):
    """Pivots and reshapes aggregated metrics into a formal summary layout."""
    # Melt long
    long_df = met_df.melt(var_name = "metrics", value_name = "perf")
    
    # Map Metric Type
    def get_metric_name(m):
        if m.startswith("accuracy_train_"): return "Accuracy_train"
        if m.startswith("accuracy_"): return "Accuracy"
        if m.startswith("sensitivity"): return "Sensitivity"
        if m.startswith("specificity"): return "Specificity"
        if m.startswith("precision"): return "Precision"
        if m.startswith("NPV"): return "NPV"
        if m.startswith("AUC_value"): return "AUROC_value"
        if m.startswith("AUPRC_value"): return "AUPRC_value"
        return None

    # Map Bound Type
    def get_bound_name(m):
        if m.endswith("_low"): return "lower"
        if m.endswith("_upp"): return "upper"
        if m.endswith("_sd"): return "sd"
        return "estimate"

    long_df['Metrics'] = long_df['metrics'].apply(get_metric_name)
    long_df['bound'] = long_df['metrics'].apply(get_bound_name)
    
    # Clean and pivot wider
    pivot_df = long_df.dropna(subset = ['Metrics'])
    summary = pivot_df.pivot_table(index = "Metrics", columns = "bound", values = "perf", aggfunc = 'first').reset_index()
    
    # Drop SD column if it exists """May remove it if standard deviation is needed"""
    if 'sd' in summary.columns:
        summary = summary.drop(columns = ['sd'])
        
    # Reorder columns to standard format
    return summary[['Metrics', 'estimate', 'lower', 'upper']]

# Function to compute the difference of performance metric between scenarios
def metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, metric, model_name):
    met_mod_list = ["accuracy_testing", "npv", "AUC_value", "PRC_val"]
    
    if metric in met_mod_list:
        if metric == "accuracy_testing":
            metric_mod = "accuracy"
        elif metric == "AUC_value":
            metric_mod = "AUROC"
        elif metric == "PRC_val":
            metric_mod = "AUPRC"
        elif metric == "npv":
            metric_mod = "NPV"
    else:
        metric_mod = metric
    
    met_df = pd.DataFrame([
        {
            f"{metric_mod} (No VL) [S1]": split1[metric],
            f"{metric_mod} (VLsymp) [S2]": split2[metric],
            f"{metric_mod} (VLdiag) [S3]": split3[metric],
            f"{metric_mod} (VLdiag+1) [S4]": split4[metric],
            f"{metric_mod} (VLdiag+2) [S5]": split5[metric]
            
        }
        for split1, split2, split3, split4, split5 in zip(model_list1, model_list2, model_list3, model_list4, model_list5)
    ])

    met_df['Model name'] = model_name
    met_df['Diff_S1_S2'] = met_df[f"{metric_mod} (No VL) [S1]"] - met_df[f"{metric_mod} (VLsymp) [S2]"]
    met_df['Diff_S1_S3'] = met_df[f"{metric_mod} (No VL) [S1]"] - met_df[f"{metric_mod} (VLdiag) [S3]"]
    met_df['Diff_S1_S4'] = met_df[f"{metric_mod} (No VL) [S1]"] - met_df[f"{metric_mod} (VLdiag+1) [S4]"]
    met_df['Diff_S1_S5'] = met_df[f"{metric_mod} (No VL) [S1]"] - met_df[f"{metric_mod} (VLdiag+2) [S5]"]

    met_df['Diff_S2_S3'] = met_df[f"{metric_mod} (VLsymp) [S2]"] - met_df[f"{metric_mod} (VLdiag) [S3]"]
    met_df['Diff_S2_S4'] = met_df[f"{metric_mod} (VLsymp) [S2]"] - met_df[f"{metric_mod} (VLdiag+1) [S4]"]
    met_df['Diff_S2_S5'] = met_df[f"{metric_mod} (VLsymp) [S2]"] - met_df[f"{metric_mod} (VLdiag+2) [S5]"]

    met_df['Diff_S3_S4'] = met_df[f"{metric_mod} (VLdiag) [S3]"] - met_df[f"{metric_mod} (VLdiag+1) [S4]"]
    met_df['Diff_S3_S5'] = met_df[f"{metric_mod} (VLdiag) [S3]"] - met_df[f"{metric_mod} (VLdiag+2) [S5]"]

    met_df['Diff_S4_S5'] = met_df[f"{metric_mod} (VLdiag+1) [S4]"] - met_df[f"{metric_mod} (VLdiag+2) [S5]"]

    return met_df

# Function to create list of performance metrics & the difference of performance metrics
def met_list_create(model_list1, model_list2, model_list3, model_list4, model_list5, model_name):
    acc_df = metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, "accuracy_testing", model_name)
    sens_df = metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, "sensitivity", model_name)
    spec_df = metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, "specificity", model_name)
    prec_df = metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, "precision", model_name)
    npv_df = metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, "npv", model_name)
    roc_df = metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, "AUC_value", model_name)
    prc_df = metric_diff(model_list1, model_list2, model_list3, model_list4, model_list5, "PRC_val", model_name)

    return {
        "acc_sim": acc_df,
        "sen_sim": sens_df,
        "spe_sim": spec_df,
        "pre_sim": prec_df,
        "npv_sim": npv_df,
        "roc_sim": roc_df,
        "prc_sim": prc_df
        }

# Function to compute win-ratio of the simulation runs
def compute_diff(model_list, model_name):
    met_sim = ["acc_sim", "sen_sim", "spe_sim", "pre_sim", "npv_sim", "roc_sim", "prc_sim"]
    rows = []

    for metric in met_sim:
        rows.append({
            "Metric": metric,
            "Total S1 - S2 < 0": np.sum(model_list[metric]["Diff_S1_S2"] < 0),
            "Total S1 - S2 > 0": np.sum(model_list[metric]["Diff_S1_S2"] > 0),
            "Total S1 - S2 = 0": np.sum(model_list[metric]["Diff_S1_S2"] == 0),
            
            "Total S1 - S3 < 0": np.sum(model_list[metric]["Diff_S1_S3"] < 0),
            "Total S1 - S3 > 0": np.sum(model_list[metric]["Diff_S1_S3"] > 0),
            "Total S1 - S3 = 0": np.sum(model_list[metric]["Diff_S1_S3"] == 0),

            "Total S1 - S4 < 0": np.sum(model_list[metric]["Diff_S1_S4"] < 0),
            "Total S1 - S4 > 0": np.sum(model_list[metric]["Diff_S1_S4"] > 0),
            "Total S1 - S4 = 0": np.sum(model_list[metric]["Diff_S1_S4"] == 0),

            "Total S1 - S5 < 0": np.sum(model_list[metric]["Diff_S1_S5"] < 0),
            "Total S1 - S5 > 0": np.sum(model_list[metric]["Diff_S1_S5"] > 0),
            "Total S1 - S5 = 0": np.sum(model_list[metric]["Diff_S1_S5"] == 0),


            "Total S2 - S3 < 0": np.sum(model_list[metric]["Diff_S2_S3"] < 0),
            "Total S2 - S3 > 0": np.sum(model_list[metric]["Diff_S2_S3"] > 0),
            "Total S2 - S3 = 0": np.sum(model_list[metric]["Diff_S2_S3"] == 0),

            "Total S2 - S4 < 0": np.sum(model_list[metric]["Diff_S2_S4"] < 0),
            "Total S2 - S4 > 0": np.sum(model_list[metric]["Diff_S2_S4"] > 0),
            "Total S2 - S4 = 0": np.sum(model_list[metric]["Diff_S2_S4"] == 0),

            "Total S2 - S5 < 0": np.sum(model_list[metric]["Diff_S2_S5"] < 0),
            "Total S2 - S5 > 0": np.sum(model_list[metric]["Diff_S2_S5"] > 0),
            "Total S2 - S5 = 0": np.sum(model_list[metric]["Diff_S2_S5"] == 0),


            "Total S3 - S4 < 0": np.sum(model_list[metric]["Diff_S3_S4"] < 0),
            "Total S3 - S4 > 0": np.sum(model_list[metric]["Diff_S3_S4"] > 0),
            "Total S3 - S4 = 0": np.sum(model_list[metric]["Diff_S3_S4"] == 0),

            "Total S3 - S5 < 0": np.sum(model_list[metric]["Diff_S3_S5"] < 0),
            "Total S3 - S5 > 0": np.sum(model_list[metric]["Diff_S3_S5"] > 0),
            "Total S3 - S5 = 0": np.sum(model_list[metric]["Diff_S3_S5"] == 0),


            "Total S4 - S5 < 0": np.sum(model_list[metric]["Diff_S4_S5"] < 0),
            "Total S4 - S5 > 0": np.sum(model_list[metric]["Diff_S4_S5"] > 0),
            "Total S4 - S5 = 0": np.sum(model_list[metric]["Diff_S4_S5"] == 0),


            "Total S2 - S1 < 0": np.sum(model_list[metric]["Diff_S1_S2"] > 0),
            "Total S2 - S1 > 0": np.sum(model_list[metric]["Diff_S1_S2"] < 0),
            "Total S2 - S1 = 0": np.sum(model_list[metric]["Diff_S1_S2"] == 0),
            
            "Total S3 - S1 < 0": np.sum(model_list[metric]["Diff_S1_S3"] > 0),
            "Total S3 - S1 > 0": np.sum(model_list[metric]["Diff_S1_S3"] < 0),
            "Total S3 - S1 = 0": np.sum(model_list[metric]["Diff_S1_S3"] == 0),

            "Total S4 - S1 < 0": np.sum(model_list[metric]["Diff_S1_S4"] > 0),
            "Total S4 - S1 > 0": np.sum(model_list[metric]["Diff_S1_S4"] < 0),
            "Total S4 - S1 = 0": np.sum(model_list[metric]["Diff_S1_S4"] == 0),

            "Total S5 - S1 < 0": np.sum(model_list[metric]["Diff_S1_S5"] > 0),
            "Total S5 - S1 > 0": np.sum(model_list[metric]["Diff_S1_S5"] < 0),
            "Total S5 - S1 = 0": np.sum(model_list[metric]["Diff_S1_S5"] == 0),


            "Total S3 - S2 < 0": np.sum(model_list[metric]["Diff_S2_S3"] > 0),
            "Total S3 - S2 > 0": np.sum(model_list[metric]["Diff_S2_S3"] < 0),
            "Total S3 - S2 = 0": np.sum(model_list[metric]["Diff_S2_S3"] == 0),

            "Total S4 - S2 < 0": np.sum(model_list[metric]["Diff_S2_S4"] > 0),
            "Total S4 - S2 > 0": np.sum(model_list[metric]["Diff_S2_S4"] < 0),
            "Total S4 - S2 = 0": np.sum(model_list[metric]["Diff_S2_S4"] == 0),

            "Total S5 - S2 < 0": np.sum(model_list[metric]["Diff_S2_S5"] > 0),
            "Total S5 - S2 > 0": np.sum(model_list[metric]["Diff_S2_S5"] < 0),
            "Total S5 - S2 = 0": np.sum(model_list[metric]["Diff_S2_S5"] == 0),


            "Total S4 - S3 < 0": np.sum(model_list[metric]["Diff_S3_S4"] > 0),
            "Total S4 - S3 > 0": np.sum(model_list[metric]["Diff_S3_S4"] < 0),
            "Total S4 - S3 = 0": np.sum(model_list[metric]["Diff_S3_S4"] == 0),

            "Total S5 - S3 < 0": np.sum(model_list[metric]["Diff_S3_S5"] > 0),
            "Total S5 - S3 > 0": np.sum(model_list[metric]["Diff_S3_S5"] < 0),
            "Total S5 - S3 = 0": np.sum(model_list[metric]["Diff_S3_S5"] == 0),


            "Total S5 - S4 < 0": np.sum(model_list[metric]["Diff_S4_S5"] > 0),
            "Total S5 - S4 > 0": np.sum(model_list[metric]["Diff_S4_S5"] < 0),
            "Total S5 - S4 = 0": np.sum(model_list[metric]["Diff_S4_S5"] == 0),
            
            "Total S1 - S1": None,
            "Total S2 - S2": None,
            "Total S3 - S3": None,
            "Total S4 - S4": None,
            "Total S5 - S5": None,

            "Total S1 - S2 Balanced": (np.sum(model_list[metric]["Diff_S1_S2"] > 0) + (np.sum(model_list[metric]["Diff_S1_S2"] == 0) / 2)),
            "Total S1 - S3 Balanced": (np.sum(model_list[metric]["Diff_S1_S3"] > 0) + (np.sum(model_list[metric]["Diff_S1_S3"] == 0) / 2)),
            "Total S1 - S4 Balanced": (np.sum(model_list[metric]["Diff_S1_S4"] > 0) + (np.sum(model_list[metric]["Diff_S1_S4"] == 0) / 2)),
            "Total S1 - S5 Balanced": (np.sum(model_list[metric]["Diff_S1_S5"] > 0) + (np.sum(model_list[metric]["Diff_S1_S5"] == 0) / 2)),
            
            "Total S2 - S3 Balanced": (np.sum(model_list[metric]["Diff_S2_S3"] > 0) + (np.sum(model_list[metric]["Diff_S2_S3"] == 0) / 2)),
            "Total S2 - S4 Balanced": (np.sum(model_list[metric]["Diff_S2_S4"] > 0) + (np.sum(model_list[metric]["Diff_S2_S4"] == 0) / 2)),
            "Total S2 - S5 Balanced": (np.sum(model_list[metric]["Diff_S2_S5"] > 0) + (np.sum(model_list[metric]["Diff_S2_S5"] == 0) / 2)),

            "Total S3 - S4 Balanced": (np.sum(model_list[metric]["Diff_S3_S4"] > 0) + (np.sum(model_list[metric]["Diff_S3_S4"] == 0) / 2)),
            "Total S3 - S5 Balanced": (np.sum(model_list[metric]["Diff_S3_S5"] > 0) + (np.sum(model_list[metric]["Diff_S3_S5"] == 0) / 2)),

            "Total S4 - S5 Balanced": (np.sum(model_list[metric]["Diff_S4_S5"] > 0) + (np.sum(model_list[metric]["Diff_S4_S5"] == 0) / 2)),

            "Total S2 - S1 Balanced": (np.sum(model_list[metric]["Diff_S1_S2"] < 0) + (np.sum(model_list[metric]["Diff_S1_S2"] == 0) / 2)),
            "Total S3 - S1 Balanced": (np.sum(model_list[metric]["Diff_S1_S3"] < 0) + (np.sum(model_list[metric]["Diff_S1_S3"] == 0) / 2)),
            "Total S4 - S1 Balanced": (np.sum(model_list[metric]["Diff_S1_S4"] < 0) + (np.sum(model_list[metric]["Diff_S1_S4"] == 0) / 2)),
            "Total S5 - S1 Balanced": (np.sum(model_list[metric]["Diff_S1_S5"] < 0) + (np.sum(model_list[metric]["Diff_S1_S5"] == 0) / 2)),

            "Total S3 - S2 Balanced": (np.sum(model_list[metric]["Diff_S2_S3"] < 0) + (np.sum(model_list[metric]["Diff_S2_S3"] == 0) / 2)),
            "Total S4 - S2 Balanced": (np.sum(model_list[metric]["Diff_S2_S4"] < 0) + (np.sum(model_list[metric]["Diff_S2_S4"] == 0) / 2)),
            "Total S5 - S2 Balanced": (np.sum(model_list[metric]["Diff_S2_S5"] < 0) + (np.sum(model_list[metric]["Diff_S2_S5"] == 0) / 2)),

            "Total S4 - S3 Balanced": (np.sum(model_list[metric]["Diff_S3_S4"] < 0) + (np.sum(model_list[metric]["Diff_S3_S4"] == 0) / 2)),
            "Total S5 - S3 Balanced": (np.sum(model_list[metric]["Diff_S3_S5"] < 0) + (np.sum(model_list[metric]["Diff_S3_S5"] == 0) / 2)),

            "Total S5 - S4 Balanced": (np.sum(model_list[metric]["Diff_S4_S5"] < 0) + (np.sum(model_list[metric]["Diff_S4_S5"] == 0) / 2)),

            "Total S1 - S1 Balanced": None,
            "Total S2 - S2 Balanced": None,
            "Total S3 - S3 Balanced": None,
            "Total S4 - S4 Balanced": None,
            "Total S5 - S5 Balanced": None
        })
    rows = pd.DataFrame(rows)
    rows['Model'] = model_name

    mapping = {"acc_sim": "Accuracy",
               "sen_sim": "Sensitivity",
               "spe_sim": "Specificity",
               "pre_sim": "Precision",
               "npv_sim": "NPV",
               "roc_sim": "AUROC value",
               "prc_sim": "AUPRC value"}
    
    rows["Metric"] = rows["Metric"].replace(mapping)
    return rows

# Function to get retrieve the simulation run with the highest AUPRC value
def get_highest_auprc(split_list):
    
    prc_df_list = []
    for i in range(100):
        prc_value = split_list[i]['PRC_val']
        temp = pd.DataFrame({
            'run': [i],
            'prc_val': [prc_value]
        })
        prc_df_list.append(temp)
    prc_df = pd.concat(prc_df_list, ignore_index = True)
    highest_prc = prc_df['prc_val'].idxmax()

    return(highest_prc)

# Function to plot the shap values plot
def shap_plot(data_list, split_list, model_type, ax = None):
    highest_round = get_highest_auprc(split_list)
    x_raw = data_list[highest_round]['train'].drop(columns = ['outcome'])
    x_encoded = x_raw.copy()
    x_encoded = x_encoded.rename(columns = {'gender': 'sex'})
    cat_cols = ['age', 'sex', 'vaccination', 'comorbidity']
    for col in cat_cols:
        x_encoded[col] = x_encoded[col].astype('category')
    
    if model_type.lower().startswith(('lasso', 'enet', 'ridge', 'elasticnet')):
        x_encoded = pd.get_dummies(x_encoded, columns = cat_cols, drop_first = False)
        shap_val = split_list[highest_round]['SHAP_values']
    elif model_type.lower().startswith(('random forest', 'rf', 'random')):
        x_encoded = pd.get_dummies(x_encoded, columns = cat_cols, drop_first = False)
        shap_val = split_list[highest_round]['SHAP_values'][:, :, 1]
    else:
        for col in x_encoded.columns:
            if isinstance(x_encoded[col].dtype, pd.CategoricalDtype):
                x_encoded[col] = x_encoded[col].cat.codes
        shap_val = split_list[highest_round]['SHAP_values']
    
    if hasattr(shap_val, 'values'):
        shap_val = shap_val.values
    shap_val = np.array(shap_val, dtype = float)

    if "vl_sim_onset" in x_encoded.columns:
        x_encoded = x_encoded.rename(columns = {"vl_sim_onset": "VL_at_onset"})
    elif "vl_sim" in x_encoded.columns and "vl_sim2" in x_encoded.columns:
        x_encoded = x_encoded.rename(columns = {"vl_sim": "VL_at_diagnosis",
                                                "vl_sim2": "VL_2_days_after_diagnosis"})
    elif "vl_sim" in x_encoded.columns:
        x_encoded = x_encoded.rename(columns = {"vl_sim": "VL_at_diagnosis"})
    else:
        x_encoded = x_encoded
    
    #plt.figure(figsize = (16, 12))
    if ax is None:
        fig, ax = plt.subplots(figsize = (16, 12))
    else:
        fig = ax.get_figure()

    shap.summary_plot(
        shap_val,
        features = x_encoded,
        feature_names = x_encoded.columns,
        show = False
    )

    #fig = plt.gcf()
    #ax = plt.gca()
    return fig, ax

# Function to compute the absolute mean SHAP values
def mean_shap(split_list, data_list, model_type):
    
    all_runs = []

    for i in range(len(split_list)):
        x_raw = data_list[i]['train'].drop(columns=['outcome']).copy()
        rename_dict = {"gender": "sex", "vl_sim_symp": "VL_at_onset", "vl_sim_onset": "VL_at_onset", "vl_sim": "VL_at_diagnosis", "vl_sim2": "VL_2_days_after_diagnosis"}
        x_encoded = x_raw.rename(columns = rename_dict)
        
        cat_cols = ['age', 'sex', 'vaccination', 'comorbidity']
        cat_cols = [col for col in cat_cols if col in x_encoded.columns]
        for col in cat_cols:
            x_encoded[col] = x_encoded[col].astype('category')
            
        if model_type.lower().startswith(('lasso', 'enet', 'ridge', 'elasticnet', 'random forest', 'rf', 'random')):
            x_encoded = pd.get_dummies(x_encoded, columns = cat_cols, drop_first = False)
        else:
            for col in x_encoded.columns:
                if isinstance(x_encoded[col].dtype, pd.CategoricalDtype):
                    x_encoded[col] = x_encoded[col].cat.codes

        feature_names = x_encoded.columns
        sv = split_list[i]['SHAP_values']
        
        if hasattr(sv, 'values'):
            sv_matrix = sv.values
        else:
            sv_matrix = sv
            
        if len(sv_matrix.shape) == 3:
            sv_matrix = sv_matrix[:, :, 1]
            
        abs_shap = np.abs(sv_matrix)
        
        df = pd.DataFrame(abs_shap, columns = feature_names)
        df['.run'] = i
        
        df_long = df.melt(id_vars = ['.run'], var_name = 'feature', value_name = 'abs_shap')
        all_runs.append(df_long)
        
    shap_long = pd.concat(all_runs, ignore_index = True)
    
    imp_avg = (
        shap_long.groupby('feature').agg(
            mean_abs_shap = ('abs_shap', lambda x: np.nanmean(x)),
            lower = ('abs_shap', lambda x: np.quantile(x, 0.025)),
            upper = ('abs_shap', lambda x: np.quantile(x, 0.975))
        ).reset_index()
    )
    
    # Sort descending by importance
    imp_avg = imp_avg.sort_values(by = 'mean_abs_shap', ascending = False).reset_index(drop = True)
    
    return imp_avg

# Function to plot the rank plot
def rank_plot(imp_avg, model):
    plot_data = imp_avg.sort_values(by = 'mean_abs_shap', ascending = True).reset_index(drop = True)
    
    fig, ax = plt.subplots(figsize = (16, 12))
    
    palette = sns.color_palette("muted", len(plot_data))
    
    for idx, row in plot_data.iterrows():
        left_err = max(0.0, row['mean_abs_shap'] - row['lower'])
        right_err = max(0.0, row['upper'] - row['mean_abs_shap'])
        
        x_err = [[left_err], [right_err]]
        
        ax.errorbar(row['mean_abs_shap'],  row['feature'], xerr = x_err,  fmt = 'none',  ecolor = palette[idx], 
                    capthick = 3.5, elinewidth = 3.5, capsize = 5, zorder = 1)
        
        ax.scatter(
            row['mean_abs_shap'], row['feature'], s = 50, edgecolors = palette[idx], facecolors = 'white', 
            linewidths = 3.5, zorder = 2)
    if model != "ElasticNet regression":
        model_name = model
    else:
        model_name = "Elastic Net regression"
    ax.set_title(f"Average SHAP feature importance across 100 runs for {model_name}", fontsize = 25, pad = 15, loc = 'left')
    ax.set_xlabel("Mean |SHAP|", fontsize = 25)
    ax.set_ylabel(None)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis = 'both', colors = 'black', labelsize = 23)
    ax.grid(False)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    return fig, ax

# Function to compute model's subgroup performance metric
def met_comp_func(df_comp, cri):
    if cri.lower() == "novl":
        temp_df = df_comp[['outcome', 'pred_noVL', 'pred_prob_noVL']].copy()
        actual = temp_df['outcome']
        pred = temp_df['pred_noVL']
        pred_prob = temp_df['pred_prob_noVL']
    elif cri.lower() == "onset":
        temp_df = df_comp[['outcome', 'pred_onset', 'pred_prob_onset']].copy()
        actual = temp_df['outcome']
        pred = temp_df['pred_onset']
        pred_prob = temp_df['pred_prob_onset']
    elif cri.lower() == "diagnosis":
        temp_df = df_comp[['outcome', 'pred_diagnosis', 'pred_prob_diagnosis']].copy()
        actual = temp_df['outcome']
        pred = temp_df['pred_diagnosis']
        pred_prob = temp_df['pred_prob_diagnosis']
    elif cri.lower() == "diagnosisplus2":
        temp_df = df_comp[['outcome', 'pred_diagnosisplus2', 'pred_prob_diagnosisplus2']].copy()
        actual = temp_df['outcome']
        pred = temp_df['pred_diagnosisplus2']
        pred_prob = temp_df['pred_prob_diagnosisplus2']
    
    mapping = {"Non severe": 0, "Severe": 1}
    actual = actual.map(mapping).astype(int)
    pred = pred.map(mapping).astype(int)

    fpr, tpr_curve, _ = roc_curve(actual, pred_prob, pos_label = 1)
    auc_val = auc(fpr, tpr_curve)
    precision_curve, recall_curve, _ = precision_recall_curve(actual, pred_prob, pos_label = 1)
    auprc_val = auc(recall_curve, precision_curve)
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels = [0, 1]).ravel()
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    sensitivity = recall_score(actual, pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "accuracy": accuracy,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "precision": precision,
        "npv": npv,
        "AUC_value": auc_val,
        "PRC_val": auprc_val
    }

def sum_metric_df(model_list):
    rows = []
    for i, result in enumerate(model_list):
        rows.append({
            'run': i + 1,
            'accuracy': result['perf']['accuracy'] * 100,
            'sensitivity': result['perf']['sensitivity'] * 100,
            'specificity': result['perf']['specificity'] * 100,
            'precision': result['perf']['precision'] * 100,
            'NPV': result['perf']['npv'] * 100,
            'AUC_value': result['perf']['AUC_value'] * 100,
            'AUPRC_value': result['perf']['PRC_val'] * 100
        })
    met_df = pd.DataFrame(rows)
    
    summary_dict = {}
    for col in met_df.columns:
        if col == 'run':
            continue
        summary_dict[f"{col}_mean"] = round(met_df[col].mean(), 4)
        summary_dict[f"{col}_sd"] = round(met_df[col].std(ddof = 1), 4)
        summary_dict[f"{col}_low"] = round(met_df[col].quantile(0.025), 4)
        summary_dict[f"{col}_upp"] = round(met_df[col].quantile(0.975), 4)
        
    met_sum = pd.DataFrame([summary_dict])
    return {"metric_df": met_df, "metric_summary": met_sum}

def met_collate_sens_func(met_df):
    long_df = met_df.melt(var_name = "metrics", value_name = "perf")
    
    def get_metric_name(m):
        if m.startswith("accuracy_"): return "Accuracy"
        if m.startswith("sensitivity"): return "Sensitivity"
        if m.startswith("specificity"): return "Specificity"
        if m.startswith("precision"): return "Precision"
        if m.startswith("NPV"): return "NPV"
        if m.startswith("AUC_value"): return "AUROC_value"
        if m.startswith("AUPRC_value"): return "AUPRC_value"
        return None

    def get_bound_name(m):
        if m.endswith("_low"): return "lower"
        if m.endswith("_upp"): return "upper"
        if m.endswith("_sd"): return "sd"
        return "estimate"

    long_df['Metrics'] = long_df['metrics'].apply(get_metric_name)
    long_df['bound'] = long_df['metrics'].apply(get_bound_name)
    
    pivot_df = long_df.dropna(subset = ['Metrics'])
    summary = pivot_df.pivot_table(index = "Metrics", columns = "bound", values = "perf", aggfunc = 'first').reset_index()
    
    if 'sd' in summary.columns:
        summary = summary.drop(columns = ['sd'])
        
    return summary[['Metrics', 'estimate', 'lower', 'upper']]

# Function to aggregate subgroup
def agg_met_func(df_for_comp, cols, criteria, sce, model):
    mapping = {"Male": "Gender\nMale", "Female": "Gender\nFemale",
               "Notfull": "Vaccination\n<2 dose", "Full": "Vaccination\n2 dose", "Booster": "Vaccination\n3 dose",
               "18-59": "Age\n18-59", "60-79": "Age\n60-79", ">=80": "Age\n≥80",
               "0": "Comorbidity\n0", "1-2": "Comorbidity\n1-2", ">=3": "Comorbidity\n≥3"}
    map_sce = {"noVL": "Without VL information", "onset": "VL at symptom onset", "diagnosis": "VL at diagnosis", "diagnosisplus2": "VL at diagnosis + 2 days"}

    temp_result = []
    for j in range(len(df_for_comp['simulation'].unique())):
        temp_df = df_for_comp[df_for_comp['simulation'] == j + 1].copy()
        temp_df = temp_df[temp_df[cols] == criteria]

        temp_result.append({'perf': met_comp_func(temp_df, sce)})
    
    met_collate = sum_metric_df(temp_result)
    final_result = met_collate_sens_func(met_collate['metric_summary']).assign(
        Model = model,
        scenario = sce,
        criteria_fil = criteria
    )
    final_result['criteria_fil'] = final_result['criteria_fil'].map(mapping)
    final_result['scenario'] = final_result['scenario'].map(map_sce)

    return final_result

# Function to make annotation text dataframe.
def make_annote_df_sens(met_df, metric, offset):
    temp_df = met_df[met_df['Metrics'].isin(['AUPRC_value', metric])][['Model', 'scenario', 'criteria_fil', 'Metrics', 'estimate', 'upper']].copy()
    wide = temp_df.pivot_table(
        index = ['Model', 'scenario', 'criteria_fil'],
        columns = 'Metrics',
        values = ['estimate', 'upper'],
        aggfunc = 'mean'
    ).reset_index()

    wide.columns = [f"{a}_{b}".rstrip("_") for a, b in wide.columns]

    ann = pd.DataFrame({
        'Model': wide['Model'],
        'scenario':wide['scenario'],
        'criteria_fil': wide['criteria_fil'],
        'Metrics': metric,
        "y": np.nanmax(
            np.vstack([
                wide["upper_AUPRC_value"],
                wide["estimate_AUPRC_value"]
            ]),
            axis = 0
        ) + offset,
        "label": wide[f"estimate_{metric}"].map(lambda x: f"{x:.1f}")
    })

    return ann

# Function to plot model's performance metric in text form in plot.
def seq_txt_plot_func(ax, ann_df, metric_name, model, prefix, color, model_to_x, x_shift = 0):
    temp = ann_df[(ann_df['Metrics'] == metric_name) & (ann_df['Model'] == model)].copy()
    temp = temp.merge(model_to_x, on = ["criteria_fil", "scenario"], how = "left")
    temp = temp.dropna(subset = ["x_pos"])
    
    for _, row in temp.iterrows():
        ax.text(
            row["x_pos"] + x_shift,
            row["y"],
            f"{prefix}\n{row['label']}%",
            color = color,
            fontsize = 9,
            ha = "center",
            va = "center",
            rotation = 360,
            fontweight = "bold"
        )

# Function to plot the subgroup model performance
def auprc_plot_sens_func(df_plot, model, ann_sen, ann_roc, ann_spe, ann_pre, ax = None):
    criteria_order = ["Gender\nMale", "Gender\nFemale",
                      "Vaccination\n<2 dose", "Vaccination\n2 dose", "Vaccination\n3 dose",
                      "Age\n60-79", "Age\n≥80",
                      "Comorbidity\n1-2", "Comorbidity\n≥3"]
    color_map = {"Without VL information": '#747474', "VL at symptom onset": '#D86ECC', "VL at diagnosis": '#348EC2', "VL at diagnosis + 2 days": '#002060'}
    scenario_order = ["Without VL information", "VL at symptom onset", "VL at diagnosis", "VL at diagnosis + 2 days"]
    
    temp_df = df_plot[(df_plot['Metrics'].isin(['AUPRC_value', 'AUROC_value', 'Sensitivity', 'Specificity', 'Precision'])) & (df_plot['Model'] == model)].copy()
    temp_df['criteria_fil'] = pd.Categorical(temp_df['criteria_fil'], categories = criteria_order, ordered = True)
    temp_df = temp_df.sort_values('criteria_fil').reset_index(drop = True)
    temp_df['scenario'] = pd.Categorical(temp_df['scenario'], categories = scenario_order, ordered = True)
    temp_df = temp_df.sort_values('scenario').reset_index(drop = True)

    prc_df = temp_df[temp_df['Metrics'] == 'AUPRC_value'].copy()
    prc_df = prc_df.sort_values("criteria_fil").reset_index(drop = True)

    if ax is None:
        fig, ax = plt.subplots(figsize = (18, 12))
    else:
        fig = ax.figure

    sns.barplot(data = prc_df, y = 'estimate', x = 'criteria_fil', orient = 'v', hue = 'scenario', errorbar = None, palette = color_map, dodge = True, order = criteria_order, width = 0.9, ax = ax)
    
    bar_rows = []
    for container, scenario in zip(ax.containers, color_map.keys()):
        scen_df = prc_df[prc_df["scenario"] == scenario].sort_values("criteria_fil")

        for bar, (_, row) in zip(container, scen_df.iterrows()):
            x = bar.get_x() + bar.get_width() / 2
            y = row["estimate"]

            bar_rows .append({
                "criteria_fil": row["criteria_fil"],
                "scenario": row["scenario"],
                "x_pos": x,
                "estimate": y,
                "lower": row["lower"],
                "upper": row["upper"]
                })
            ax.errorbar(y = y, x = x, yerr = [[y - row["lower"]], [row["upper"] - y]], fmt = 'none', color = 'black', elinewidth = 1.5, capsize = 5)
    model_to_x = pd.DataFrame(bar_rows)
    
    seq_txt_plot_func(ax, ann_sen, "Sensitivity", model, "Sens", "#196824", model_to_x, x_shift = 0)
    seq_txt_plot_func(ax, ann_roc, "AUROC_value", model, "ROC", "#9e9ac8", model_to_x, x_shift = 0)
    seq_txt_plot_func(ax, ann_spe, "Specificity", model, "Spec", "#FA766E", model_to_x, x_shift = 0)
    seq_txt_plot_func(ax, ann_pre, "Precision", model, "Prec", "#1c9099", model_to_x, x_shift = 0)

    ax.set_ylabel("AUPRC (%)", fontsize = 18)
    ax.set_xlabel("")

    ax.set_ylim(0, 120)
    ax.set_yticks(np.arange(0, 110, 10))

    ax.grid(False)
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis = "y", colors = "black", labelsize = 18)
    ax.tick_params(axis = "x", colors = "black", labelsize = 18)

    ax.legend(title = None, fontsize = 14, frameon = False, ncol = 4, loc = "lower center",  bbox_to_anchor = (0.5, 1.02))

    if model != "ElasticNet regression":
        model_name = model
    else:
        model_name = "Elastic Net regression"
    ax.set_title(f"Performance of {model_name} with stratified characteristics", fontsize = 18, fontweight = "bold", y = 1.08)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()

    return ax

# Function to plot boxplot for model's acurracy in training-testing sets comparison
def plot_acc_func(df_plot, model_check, ax = None):
    color_map = {"Training": '#d95f02', "Testing": '#1b9e77'}
    scenario_order = ["Without VL information", "VL at symptom onset", "VL at diagnosis", "VL at diagnosis + 2 days"]

    temp = df_plot[df_plot['model'] == model_check].copy()
    #if ""

    temp['scenario'] = pd.Categorical(temp['scenario'], categories = scenario_order, ordered = True)
    temp = temp.sort_values('scenario').reset_index(drop = True)

    temp = temp.melt(
        id_vars = ["run", "scenario"],
        value_vars=["accuracy_train", "accuracy_test"],
        var_name = "Dataset",
        value_name = "Accuracy"
    )

    temp["Dataset"] = temp["Dataset"].map({
        "accuracy_train": "Training",
        "accuracy_test": "Testing"
    })

    if ax is None:
        fig, ax = plt.subplots(figsize = (14, 13))
    
    sns.boxplot(data = temp, x = 'scenario', y = 'Accuracy', hue = 'Dataset', palette = color_map, showcaps = False, showfliers = False, ax = ax)
    
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)", fontsize = 40)
    
    if model_check != "ElasticNet regression":
        model_name = model_check
    else:
        model_name = "Elastic Net regression"
    ax.set_title(f"Accuracy in the Training and Tests Sets\nacross 100 Analyses for {model_name}", fontsize = 32)

    if model_check == "LightGBM":
        ax.set_ylim(50, 100)
        ax.set_yticks(np.arange(50, 105, 10))
    else:
        ax.set_ylim(75, 100)
        ax.set_yticks(np.arange(75, 105, 5))

    ax.grid(False)
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)

    ax.tick_params(axis = "y", colors = "black", labelsize = 35)
    ax.tick_params(axis = "x", colors = "black", labelsize = 28, rotation = 45)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.legend(fontsize = 30, frameon = False, ncol = 2, loc = "upper right")
    plt.tight_layout()
    
    return ax

def make_annote_df_seq(met_df, metric, offset):
    temp_df = met_df[met_df['Metrics'].isin(['AUPRC value', metric])][['Model', 'day_seq', 'Metrics', 'estimate', 'upper']].copy()
    wide = temp_df.pivot_table(
        index = ['Model', 'day_seq'],
        columns = 'Metrics',
        values = ['estimate', 'upper'],
        aggfunc = 'mean'
    ).reset_index()

    wide.columns = [f"{a}_{b}".rstrip("_") for a, b in wide.columns]

    ann = pd.DataFrame({
        'Model': wide['Model'],
        'day_seq':wide['day_seq'],
        'Metrics': metric,
        "y": np.nanmax(
            np.vstack([
                wide["upper_AUPRC value"],
                wide["estimate_AUPRC value"]
            ]),
            axis = 0
        ) + offset,
        "label": wide[f"estimate_{metric}"].map(lambda x: f"{x:.1f}")
    })

    return ann

def seq_txt_plot_func2(ax, ann_df, metric_name, model, prefix, color, model_to_x, x_shift = 0):
    temp = ann_df[(ann_df['Metrics'] == metric_name) & (ann_df['Model'] == model)].copy()
    temp["day_seq"] = temp["day_seq"].astype(str)
    temp["x_pos"] = temp["day_seq"].map(model_to_x)
    temp = temp.dropna(subset = ["x_pos"])
    
    for _, row in temp.iterrows():
        ax.text(
            row["x_pos"] + x_shift,
            row["y"],
            f"{prefix}\n{row['label']}%",
            color = color,
            fontsize = 14,
            ha = "center",
            va = "center",
            rotation = 360,
            fontweight = "bold"
        )

# Function to plot model's performance metrics with subsequent VL after diagnosis
def auprc_plot_seq_func(df_plot, model, ann_sen, ann_roc, ann_spe, ann_pre, ax = None):
    day_seq_order = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    #model_order = ['XGBoost', 'LightGBM', 'Random Forest', 'Ridge regression', 'Lasso regression', 'ElasticNet regression']
    color_map = {"0": '#7f3b08', "1": '#b35806', "2": '#e08214', "3": '#fdb863', "4": '#fee0b6', "5": '#f7f7f7',
                 "6": '#d8daeb', "7": '#b2abd2', "8": '#8073ac', "9": '#542788', "10": '#2d004b'}
    
    temp_df = df_plot[(df_plot['Metrics'].isin(['AUPRC value', 'AUROC_value', 'Sensitivity', 'Specificity', 'Precision'])) & (df_plot['Model'] == model)].copy()
    temp_df["day_seq"] = temp_df["day_seq"].astype(str)
    temp_df['day_seq'] = pd.Categorical(temp_df['day_seq'], categories = day_seq_order, ordered = True)
    temp_df = temp_df.sort_values('day_seq').reset_index(drop = True)

    prc_df = temp_df[temp_df['Metrics'] == 'AUPRC value'].copy()
    prc_df = prc_df.sort_values("day_seq").reset_index(drop = True)

    if ax is None:
        fig, ax = plt.subplots(figsize = (16, 12))
    else:
        fig = ax.figure

    sns.barplot(data = prc_df, y = 'estimate', x = 'day_seq', orient = 'v', hue = 'day_seq', errorbar = None, palette = color_map, dodge = False, order = day_seq_order, legend = False, ax = ax, gap = 1.5)
    
    x_pos = np.arange(len(day_seq_order))
    model_to_x = dict(zip(day_seq_order, x_pos))
    prc_df['x_pos'] = prc_df['day_seq'].map(model_to_x)

    yerr = np.vstack([prc_df['estimate'].to_numpy() - prc_df['lower'].to_numpy(),
                     prc_df['upper'].to_numpy() - prc_df['estimate'].to_numpy()])
    ax.errorbar(y = prc_df['estimate'], x = prc_df['x_pos'], yerr = yerr, fmt = 'none', color = 'black', elinewidth = 1.5, capsize = 5)
    
    seq_txt_plot_func2(ax, ann_roc, "AUROC_value", model, "AUROC", "#9e9ac8", model_to_x, x_shift = 0)
    seq_txt_plot_func2(ax, ann_spe, "Specificity", model, "Specificity", "#FA766E", model_to_x, x_shift = 0)
    seq_txt_plot_func2(ax, ann_pre, "Precision", model, "Precision", "#1c9099", model_to_x, x_shift = 0)
    seq_txt_plot_func2(ax, ann_sen, "Sensitivity", model, "Sensitivity", "#196824", model_to_x, x_shift = 0)

    ax.set_ylabel("AUPRC (%)", fontsize = 28)
    ax.set_xlabel("")

    ax.set_ylim(0, 95)
    ax.set_yticks(np.arange(0, 90, 10))

    ax.grid(False)
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis = "y", colors = "black", labelsize = 23)
    ax.tick_params(axis = "x", colors = "black", labelsize = 23)

    if model != "ElasticNet regression":
        model_name = model
    else:
        model_name = "Elastic Net regression"
    ax.set_title(f"Performance of {model_name} with sequential VL at x day after diagnosis", fontsize = 20, fontweight = "bold")

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()

    return ax

# Function to compute mean difference in model performance
def compute_mean_diff(model_list, model_name):
    met_sim = ["acc_sim", "sen_sim", "spe_sim", "pre_sim", "npv_sim", "roc_sim", "prc_sim"]
    rows = []

    for metric in met_sim:
        rows.append({
            "Metric": metric,
            "Mean S1 - S2": np.mean(model_list[metric]["Diff_S1_S2"]),
            "Mean S1 - S3": np.mean(model_list[metric]["Diff_S1_S3"]),
            "Mean S1 - S4": np.mean(model_list[metric]["Diff_S1_S4"]),
            "Mean S1 - S5": np.mean(model_list[metric]["Diff_S1_S5"]),
            
            "Mean S2 - S3": np.mean(model_list[metric]["Diff_S2_S3"]),
            "Mean S2 - S4": np.mean(model_list[metric]["Diff_S2_S4"]),
            "Mean S2 - S5": np.mean(model_list[metric]["Diff_S2_S5"]),
            
            "Mean S3 - S4": np.mean(model_list[metric]["Diff_S3_S4"]),
            "Mean S3 - S5": np.mean(model_list[metric]["Diff_S3_S5"]),
            
            "Mean S4 - S5": np.mean(model_list[metric]["Diff_S4_S5"]),


            "Lower S1 - S2": np.quantile(model_list[metric]["Diff_S1_S2"], 0.025),
            "Lower S1 - S3": np.quantile(model_list[metric]["Diff_S1_S3"], 0.025),
            "Lower S1 - S4": np.quantile(model_list[metric]["Diff_S1_S4"], 0.025),
            "Lower S1 - S5": np.quantile(model_list[metric]["Diff_S1_S5"], 0.025),
            
            "Lower S2 - S3": np.quantile(model_list[metric]["Diff_S2_S3"], 0.025),
            "Lower S2 - S4": np.quantile(model_list[metric]["Diff_S2_S4"], 0.025),
            "Lower S2 - S5": np.quantile(model_list[metric]["Diff_S2_S5"], 0.025),
            
            "Lower S3 - S4": np.quantile(model_list[metric]["Diff_S3_S4"], 0.025),
            "Lower S3 - S5": np.quantile(model_list[metric]["Diff_S3_S5"], 0.025),
            
            "Lower S4 - S5": np.quantile(model_list[metric]["Diff_S4_S5"], 0.025),


            "Upper S1 - S2": np.quantile(model_list[metric]["Diff_S1_S2"], 0.975),
            "Upper S1 - S3": np.quantile(model_list[metric]["Diff_S1_S3"], 0.975),
            "Upper S1 - S4": np.quantile(model_list[metric]["Diff_S1_S4"], 0.975),
            "Upper S1 - S5": np.quantile(model_list[metric]["Diff_S1_S5"], 0.975),
            
            "Upper S2 - S3": np.quantile(model_list[metric]["Diff_S2_S3"], 0.975),
            "Upper S2 - S4": np.quantile(model_list[metric]["Diff_S2_S4"], 0.975),
            "Upper S2 - S5": np.quantile(model_list[metric]["Diff_S2_S5"], 0.975),
            
            "Upper S3 - S4": np.quantile(model_list[metric]["Diff_S3_S4"], 0.975),
            "Upper S3 - S5": np.quantile(model_list[metric]["Diff_S3_S5"], 0.975),
            
            "Upper S4 - S5": np.quantile(model_list[metric]["Diff_S4_S5"], 0.975), 
            

            "Mean S2 - S1": np.mean(-model_list[metric]["Diff_S1_S2"]),
            "Mean S3 - S1": np.mean(-model_list[metric]["Diff_S1_S3"]),
            "Mean S4 - S1": np.mean(-model_list[metric]["Diff_S1_S4"]),
            "Mean S5 - S1": np.mean(-model_list[metric]["Diff_S1_S5"]),

            "Mean S3 - S2": np.mean(-model_list[metric]["Diff_S2_S3"]),
            "Mean S4 - S2": np.mean(-model_list[metric]["Diff_S2_S4"]),
            "Mean S5 - S2": np.mean(-model_list[metric]["Diff_S2_S5"]),


            "Mean S4 - S3": np.mean(-model_list[metric]["Diff_S3_S4"]),
            "Mean S5 - S3": np.mean(-model_list[metric]["Diff_S3_S5"]),

            "Mean S5 - S4": np.mean(-model_list[metric]["Diff_S4_S5"]),
            

            "Lower S2 - S1": np.quantile(-model_list[metric]["Diff_S1_S2"], 0.025),
            "Lower S3 - S1": np.quantile(-model_list[metric]["Diff_S1_S3"], 0.025),
            "Lower S4 - S1": np.quantile(-model_list[metric]["Diff_S1_S4"], 0.025),
            "Lower S5 - S1": np.quantile(-model_list[metric]["Diff_S1_S5"], 0.025),

            "Lower S3 - S2": np.quantile(-model_list[metric]["Diff_S2_S3"], 0.025),
            "Lower S4 - S2": np.quantile(-model_list[metric]["Diff_S2_S4"], 0.025),
            "Lower S5 - S2": np.quantile(-model_list[metric]["Diff_S2_S5"], 0.025),


            "Lower S4 - S3": np.quantile(-model_list[metric]["Diff_S3_S4"], 0.025),
            "Lower S5 - S3": np.quantile(-model_list[metric]["Diff_S3_S5"], 0.025),

            "Lower S5 - S4": np.quantile(-model_list[metric]["Diff_S4_S5"], 0.025),


            "Upper S2 - S1": np.quantile(-model_list[metric]["Diff_S1_S2"], 0.975),
            "Upper S3 - S1": np.quantile(-model_list[metric]["Diff_S1_S3"], 0.975),
            "Upper S4 - S1": np.quantile(-model_list[metric]["Diff_S1_S4"], 0.975),
            "Upper S5 - S1": np.quantile(-model_list[metric]["Diff_S1_S5"], 0.975),

            "Upper S3 - S2": np.quantile(-model_list[metric]["Diff_S2_S3"], 0.975),
            "Upper S4 - S2": np.quantile(-model_list[metric]["Diff_S2_S4"], 0.975),
            "Upper S5 - S2": np.quantile(-model_list[metric]["Diff_S2_S5"], 0.975),


            "Upper S4 - S3": np.quantile(-model_list[metric]["Diff_S3_S4"], 0.975),
            "Upper S5 - S3": np.quantile(-model_list[metric]["Diff_S3_S5"], 0.975),

            "Upper S5 - S4": np.quantile(-model_list[metric]["Diff_S4_S5"], 0.975),


            "Mean S1 - S1": None,
            "Mean S2 - S2": None,
            "Mean S3 - S3": None,
            "Mean S4 - S4": None,
            "Mean S5 - S5": None
        })
    rows = pd.DataFrame(rows)
    rows['Model'] = model_name

    mapping = {"acc_sim": "Accuracy",
               "sen_sim": "Sensitivity",
               "spe_sim": "Specificity",
               "pre_sim": "Precision",
               "npv_sim": "NPV",
               "roc_sim": "AUROC value",
               "prc_sim": "AUPRC value"}
    
    rows["Metric"] = rows["Metric"].replace(mapping)
    return rows