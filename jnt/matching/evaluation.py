from glob import glob
from pandas import read_excel, read_csv
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    accuracy_score, roc_auc_score, classification_report, precision_recall_fscore_support

from pandas import read_excel, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    accuracy_score, roc_auc_score, classification_report, precision_recall_fscore_support
from collections import defaultdict
import numpy as np


NAME_PERSON_OK = True


def load_matches(match_fpath):
    match_df = read_csv(match_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)

    matches = defaultdict(dict)
    for i, row in match_df.iterrows():
        matches[make_url(row.sense_i)][row.sense_j] = row.sim

    return matches


def global_threshold_pr(golden_fpath, match_fpath, random=False):
    golden_df = read_excel(golden_fpath, sheetname='data')
    matches = load_matches(match_fpath)

    for i, row in golden_df.iterrows():
        golden_df.loc[i, "sim"] = 0.0
        if row.sense_i in matches and row.sense_j in matches[row.sense_i]:
            golden_df.loc[i, "sim"] = matches[row.sense_i][row.sense_j]

    output_fpath = golden_fpath + "-sim.csv"
    golden_df.to_csv(output_fpath, sep="\t", encoding="utf-8", float_format='%.6f', index=False)
    #print "Evaluation data with predictions:", output_fpath

    s = {}
    if random:
        sim = np.random.random(len(golden_df.sim))
    else:
        sim = golden_df.sim
    s["precision"], s["recall"], s["threshold"] = precision_recall_curve(golden_df.match, sim)
    s["average_precision"] = average_precision_score(golden_df.match, sim)

    return s


def print_result(match_name, match_fpath, golden_fpath, random=False):
    s = global_threshold_pr(golden_fpath, match_fpath, random)
    plt.plot(s["recall"], s["precision"], label=match_name + " (AUC = %.2f)" % s["average_precision"], linewidth=2)
    print match_name, "AUC", s["average_precision"]




def make_url(babelnet_id):
    return u"http://babelnet.org/synset?word="+ babelnet_id + u"&lang=EN&details=1"


def fix_00(precision, recall):

    result = []
    for r,p in zip(recall, precision):
        if r == .0 and p == .0: continue  #result.append((.0,1.))
        else: result.append((r,p))

    r, p = zip(*result)
    return p, r


def evaluate_match(golden_fpath, match_fpath, threshold=""):
    golden_df = read_excel(golden_fpath, sheetname='data')
    match_df = read_csv(match_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    matches = defaultdict(dict)
    
    for i, row in match_df.iterrows():
        matches[make_url(row.sense_i)][row.sense_j] = row.sim

    for i, row in golden_df.iterrows():
        if row.sense_i in matches and row.sense_j in matches[row.sense_i]:
            golden_df.loc[i, "match_predict"] = 1
        else:
            golden_df.loc[i, "match_predict"] = 0

    precision, recall, fmeasure, support = precision_recall_fscore_support(golden_df.match, golden_df.match_predict)

    r = {}
    r["precision"] = precision[1]
    r["recall"] = recall[1]
    r["fmeasure"] = fmeasure[1]
    r["support"] = support[1]
    r["threshold"] = threshold
    r["file"] = match_fpath
    
    return r
 
    
def get_pr(results):

    recall = []
    precision = []

    for r in results:
        precision.append(r["precision"])
        recall.append(r["recall"])

    return precision, recall


def get_precision_recall(golden_fpath, match_pattern):

    results = []

    for match_fpath in glob(match_pattern):
        print ".",
        results.append(evaluate_match(golden_fpath, match_fpath, threshold=0.80))

    p, r = get_pr(results)
    p, r = fix_00(p, r)
    p, r = zip(*sorted(zip(p,r), reverse=True))

    return p, r

