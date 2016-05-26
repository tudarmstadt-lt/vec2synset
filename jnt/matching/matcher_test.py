# # RUN ON 1100 WORDS
#
# from jnt.matching.synset_fetchers import BabelNet, SenseClusters, BABELNET_KEYS
# from jnt.matching.matcher import Matcher
#
# BABELNET_DIR = "/Users/alex/tmp/matching/babelnet/"
# ADAGRAM_DDT = "/Users/alex/tmp/matching/agagram-hugemodel-ambigous1100.csv.out"
# voc_fpath = "/Users/alex/work/joint/src/data/ambigous-words-mine.csv"
#
# babelnet = BabelNet(babelnet_keys=BABELNET_KEYS, babelnet_dir=BABELNET_DIR)
# adagram = SenseClusters(ADAGRAM_DDT, strip_dst_senses=False, normalized_bow=True)
#
# m = Matcher(babelnet, adagram)
# m.match_file(voc_fpath, voc_fpath + "-match.csv")
# #babelnet.save()


# from jnt.matching.synset_fetchers import BabelNet, SenseClusters, BABELNET_KEYS
# from jnt.matching.matcher import Matcher
# from math import ceil
# from pandas import read_csv
# from collections import defaultdict
# from numpy import std
#
# BABELNET_DIR = "/Users/alex/tmp/matching/babelnet/"
# ADAGRAM_DDT = "/Users/alex/tmp/matching/ddt-adagram-ukwac+wacky.csv.gz.voc.out"
# voc_fpath = "/Users/alex/work/joint/src/data/ambigous-words-mine.csv"
#
# babelnet = BabelNet(babelnet_keys=BABELNET_KEYS, babelnet_dir=BABELNET_DIR)
# adagram = SenseClusters(ADAGRAM_DDT, normalized_bow=True)
#
# m = Matcher(babelnet, adagram)
# res = defaultdict(list)
#
# for p in range(0,100,5):
#     match_fpath = voc_fpath + "-match-p%d.csv" % p
#     m.match_file(voc_fpath, match_fpath, threshold_percentile=p)
#     df = read_csv(match_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
#     res["matches"].append(len(df))
#     res["words"].append(len(set(df.word.values)))
#     res["matches/words"].append(float(len(df)) / len(set(df.word.values)))
#     res["p"].append(p)
#     l = [len(rows) for word, rows in df.groupby(["word"])]
#     res["std"].append(std(l))


from jnt.matching.synset_fetchers import BabelNet, SenseClusters, BABELNET_KEYS
from jnt.matching.matcher import Matcher

BABELNET_DIR = "/Users/alex/tmp/matching/babelnet-eval/"
ADAGRAM_DDT = "/Users/alex/tmp/matching/ddt-adagram-ukwac+wacky.csv.gz.voc.out"
voc_fpath = "/Users/alex/work/joint/src/data/ambigous-words-mine.csv"
freq_fpath = "" # "/Users/alex/tmp/st/word-freq-t10.csv"

from time import time
tic = time()
babelnet = BabelNet(babelnet_keys=BABELNET_KEYS, babelnet_fpath=BABELNET_DIR, freq_fpath=freq_fpath, divide_by_freq=False)
print "BabelNet load time:", time()-tic

tic = time()
adagram = SenseClusters(ADAGRAM_DDT, normalized_bow=True)
print "SenseClusters load time:", time()-tic

m = Matcher(babelnet, adagram)
m.match_file(voc_fpath, voc_fpath + "-match.csv")
