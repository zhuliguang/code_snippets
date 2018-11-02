###
#  @Author: Leon Zhu
#  @Since: 20th Aug 2018
#  @last: 20th Aug 2018
###
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import cophenet,fcluster

import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
logging.getLogger().setLevel(logging.INFO)


# @contextmanager
# def timer(title):
#     t0 = time.time()
#     yield
#     print("{} - done in {:.0f}s".format(title, time.time() - t0))

def cardinality_screening(df, nan_as_category=True, max_levels=20):
    logging.info(">>> Analysing cardinality ... ")
    def get_top_95_dist(series):
        budget=1.0
        dist = series.value_counts(normalize=True, dropna=False)
        include_level = []
        for idx, count_pst in dist.iteritems():
            budget-=count_pst
            if budget >= 0.05:
                include_level.append(idx)
            else:
                break
        return include_level

    ones = {}
    manys = {}
    for col in df.select_dtypes('object').columns:
        effective_values = get_top_95_dist(df[col])
        if len(effective_values) == 1:
            ones[col] = effective_values          
            continue
        if len(effective_values) >= max_levels:
            manys[col] = effective_values
            continue
        all_values = df[col].unique().tolist()
        for v in all_values:
            if v not in effective_values:
                df[col].replace(v, 'other', inplace=True)
                logging.info("\tValue '{}' in [{}] has been replaced by 'other'".format(v,col))
    for o in ones:
        logging.info("\t[{}] has been excluded becasue it has 1 level {}".format(o, ones[o]))
    for m in manys:
        logging.info("\t[{}] has been excluded becasue the number of its effective value is more than {}".format(m,max_levels))

    return df[[col for col in df.columns if (col not in ones and col not in manys)]]    
    
def one_hot_encoder(df, nan_as_category=True):
    logging.info(">>> Creating one hot encoding for categorical columns")
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    return df

def target_skewness(df, target, positive_token=1):
    logging.info(">>> Calculating skewness of target ...")
    pct = df[target].value_counts()[positive_token] / len(df)
    logging.info("\tThere are {0:.1%} of positive example".format(pct))

def nulls_num(df):
    logging.info(">>> Analysing null values ...")
    nums = df.select_dtypes(include=np.number)
    nulls = {}
    for col in nums.columns:
        nulls[col] = round(100*len(df[df[col].isnull()][col]) / len(df[col]), 2)

    res = pd.DataFrame(sorted(nulls.items(), key=lambda x: x[1], reverse=True))
    res.columns = ['colName','missing']
    res = res.set_index('colName')
    return res

def collinearity(df, target, threshold=0.2):
    logging.info(">>> Analysing collinearity ... ")
    nums = df.select_dtypes(include=np.number)

    logging.info(">>> Calculating pairwise correlation matrix ...")
    corr = abs(nums.corr())
    dis_matrix=(1-corr).fillna(0).as_matrix()
    flat_dis = squareform(dis_matrix)

    logging.info(">>> Clustering ... ")
    Y = sch.linkage(flat_dis, method='average')
    #inc = [col for col in corr.columns if col != target]
    #corr = corr.loc[inc]
    linear_target = corr[target]
    #corr = corr[inc]
    cluster_assignment = {}
    clstr = fcluster(Y, t=threshold, criterion='distance')

    logging.info(">>> Assigning clusters ...")
    for col, c in zip(corr.index, clstr):
        cluster_assignment[col] = [c, linear_target.loc[col]]
    res = pd.DataFrame(cluster_assignment).T
    res.columns = ['cluster','linear_correlation']
    res = res.sort_values('cluster')
    return res

def pick_values(df):
    logging.info(">>> Suggesting features to pick from")
    new_df = df.copy()
    new_df['compound'] = new_df['linear_correlation'] * (100-new_df['missing'])
    selected = pd.DataFrame(new_df.groupby('cluster')['compound'].idxmax())
    selected['Pick_me'] = 'Y'
    selected = selected.set_index('compound')
    new_df = new_df.join(selected).sort_values("cluster")
    return new_df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path',help='The path to training data file')
    parser.add_argument('target',help='Col name of modelling target')
    parser.add_argument('output',help='Output file path')
    parser.add_argument('-s','--delimiter', default=",", help='delimiter of data file. eg "," or "\\t". Default ","')
    parser.add_argument('-l','--max_levels',default=10, help='maximum number of levels allowed in categorical features')
    args = parser.parse_args()
    
    logging.info(">>> Loading data ...")
    df = pd.read_csv(args.path, delimiter=args.delimiter)
    logging.info(">>> data loaded")

    target_skewness(df, args.target, positive_token=1)
    nulls = nulls_num(df)
    card = cardinality_screening(df)
    ohe = one_hot_encoder(card)
    coll = collinearity(ohe, args.target)

    result = pick_values(pd.concat([coll, nulls], axis=1, sort=False))
    result.drop(args.target,axis=0)
    result.to_csv(args.output)
    print(result)
