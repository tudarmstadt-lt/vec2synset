from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from numpy import percentile
from collections import defaultdict
import operator
from traceback import format_exc
from jnt.common import take
from jnt.matching.synset_fetchers import DailyLimitException
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import binarize
from jnt.matching.synset_fetchers import filter_ddt_by_voc
from jnt.matching.synset_fetchers import BabelNet, SenseClusters, BABELNET_KEYS
import codecs
from os.path import join
from jnt.wn import WordNetOffsets
from pandas import read_csv, merge, Series


VERBOSE = False
WORDS_SEP = "\t"
CLUSTER_SIZE = 15
THRESHOLD = 0.05

# Meta-parameters of the method
NORM = "l2"
BOW_NORM = "unit"  # "binary", "unit", "tfidf"
PERCENTILE = 80
MIN_SENSE_PROB = 0.05


def create_mapping(babelnet_dir, adagram_fpath, match_fpath, threshold=THRESHOLD):
    print ">>> write babelnet vocabulary"
    babelnet = BabelNet(babelnet_keys=BABELNET_KEYS, babelnet_fpath=babelnet_dir)
    babelnet_voc_fpath = join(babelnet_dir, "voc.csv")
    with codecs.open(babelnet_voc_fpath, "w", "utf-8") as out:
        out.write("word\n")
        for w in babelnet.data: out.write("%s\n" % w)
    print "BabelNet vocabulary:", babelnet_voc_fpath

    print ">>> make a subset of adagram"
    adagram_voc_fpath = adagram_fpath + "-voc.csv"
    filter_ddt_by_voc(adagram_fpath, babelnet_voc_fpath, adagram_voc_fpath)

    print ">>> calculare similarities between all words"
    adagram = SenseClusters(adagram_voc_fpath, normalized_bow=True)
    m = Matcher(babelnet, adagram)
    match_all_fpath = match_fpath + "-all.csv"
    m.match_file(babelnet_voc_fpath, match_all_fpath, threshold_percentile=0.0)

    print ">>> threshold the similarity"
    match_df = read_csv(match_all_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    all_words_num = len(set(match_df.word))
    candidates_num = len(match_df)
    match_df = match_df[match_df.sim >= threshold]
    print "# of mapping candidates", candidates_num
    print "# of mappings:", len(match_df)
    print "# total words", all_words_num
    print "# mapped words", len(set(match_df.word))
    raw_match_fpath =  match_fpath + "-raw.csv"
    match_df.to_csv(raw_match_fpath, sep="\t", encoding="utf-8", float_format='%.3f', index=False)
    print "Raw mapping file:", raw_match_fpath

    print ">>> map to wordnet and reformat"
    wordnet = WordNetOffsets()
    bn2wn = babelnet.wn_mapping()
    df = read_csv(raw_match_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    df["wordnet_id"] = Series("", df.index)
    df["wordnet_cluster"] = Series("", df.index)

    for i, row in df.iterrows():
        bn_id = row.sense_i

        if row.word in bn2wn and bn_id in bn2wn[row.word] and len(bn2wn[row.word][bn_id]) > 0:
            wn_id = bn2wn[row.word][bn_id][0]
            df.loc[i,"wordnet_id"] = wn_id
            s = wordnet.get(wn_id)
            if s is not None:
                df.loc[i,"wordnet_cluster"] = s.definition() + u" " + ". ".join(s.examples())
            if len(bn2wn[row.word][bn_id]) > 1: print "*",

    df.columns = [u'word', u'babelnet_id', u"adagram_id", u"babelnet_adagram_sim",
                  u"babelnet_cluster", u"adagram_cluster", u"wordnet_id", "wordnet_cluster"]
    df = df[[u'word', u'babelnet_id', u"adagram_id", u"wordnet_id", u"babelnet_adagram_sim",
                  u"babelnet_cluster", u"adagram_cluster", u"wordnet_cluster"]]
    df.to_csv(match_fpath, sep="\t", encoding="utf-8", float_format='%.3f', index=False)

    print "Final mapping:", match_fpath



class Matcher(object):
    def __init__(self, synset_fetcher_1, synset_fetcher_2):
        self._fetcher1 = synset_fetcher_1
        self._fetcher2 = synset_fetcher_2

    def generate_non_matching(self, words_fpath, output_fpath, continue_locally=True, threshold_percentile=PERCENTILE):

        with codecs.open(output_fpath, "w", "utf-8") as output_file:
            print >> output_file, "word\tsense_i\tsense_j\tsim\tsense_i_cluster\tsense_j_cluster"

            df = read_csv(words_fpath, encoding='utf-8', delimiter=WORDS_SEP, error_bad_lines=False)
            for i, row in df.iterrows():
                try:
                    senses1 = self._fetcher1.get_senses(row.word)
                    senses2 = self._fetcher2.get_senses(row.word)
                    res = self._match(row.word, senses1, senses2, q=threshold_percentile)

                    for sid1, bow1 in senses1:
                        for sid2, bow2 in senses2:
                            if sid1 not in res and sid2 not in res[sid1]:
                                cluster1 = ','.join(take(CLUSTER_SIZE,[x[0] for x in sorted(self._fetcher1.get_cluster(row.word, sid1).items(), reverse=True, key=operator.itemgetter(1))]))
                                cluster2 = ','.join(take(CLUSTER_SIZE,[x[0] for x in sorted(self._fetcher2.get_cluster(row.word, sid2).items(), reverse=True, key=operator.itemgetter(1))]))
                                output_file.write("%s\t%s\t%s\t%.2f\t%s\t%s\n" % (
                                    row.word, sid1, sid2, 0.0, cluster1, cluster2))
                except KeyboardInterrupt:
                    print "Keyboard interrupt"
                    return
                except DailyLimitException:
                    if continue_locally:
                        print "Skipping due to API limit:", row.word
                        continue
                    else:
                        print "BabelNet API daily limit reached"
                        return
                except:
                    print "Error:", row
                    print format_exc()
        print "Matched senses:", output_fpath

        return output_fpath


    def match_file(self, words_fpath, output_fpath, continue_locally=True, threshold_percentile=PERCENTILE):

        with codecs.open(output_fpath, "w", "utf-8") as output_file:
            print >> output_file, "word\tsense_i\tsense_j\tsim\tsense_i_cluster\tsense_j_cluster"

            df = read_csv(words_fpath, encoding='utf-8', delimiter=WORDS_SEP, error_bad_lines=False)
            for i, row in df.iterrows():
                try:
                    senses1 = self._fetcher1.get_senses(row.word, min_prob=MIN_SENSE_PROB)
                    senses2 = self._fetcher2.get_senses(row.word, min_prob=MIN_SENSE_PROB)
                    res = self._match(row.word, senses1, senses2, q=threshold_percentile)

                    for sid1 in res:
                        for sid2, sim in sorted(res[sid1].items(), key=operator.itemgetter(1), reverse=True):
                            cluster1 = ','.join(take(CLUSTER_SIZE,[x[0] for x in sorted(self._fetcher1.get_cluster(row.word, sid1).items(), reverse=True, key=operator.itemgetter(1))]))
                            cluster2 = ','.join(take(CLUSTER_SIZE,[x[0] for x in sorted(self._fetcher2.get_cluster(row.word, sid2).items(), reverse=True, key=operator.itemgetter(1))]))
                            output_file.write("%s\t%s\t%s\t%.6f\t%s\t%s\n" % (
                                row.word, sid1, sid2, sim, cluster1, cluster2))
                except KeyboardInterrupt:
                    print "Keyboard interrupt"
                    return
                except DailyLimitException:
                    if continue_locally:
                        print "Skipping due to API limit:", row.word
                        continue
                    else:
                        print "BabelNet API daily limit reached"
                        return
                except:
                    print "Error:", row
                    print format_exc()
        print "Matched senses:", output_fpath

        return output_fpath

    def _match(self, word, senses1, senses2, q=PERCENTILE, verbose=False, norm=BOW_NORM):
        res = defaultdict(dict)

        bows1 = [s[1] for s in senses1]
        for b in bows1: del b[word]
        bows2 = [s[1] for s in senses2]
        for b in bows2: del b[word]
        senses1 = [(s[0], s[1]) for s in senses1]
        senses2 = [(s[0], s[1]) for s in senses2]
        bows = bows1 + bows2
        if len(bows) <= 0: return res

        _dv = DictVectorizer(separator='=', sparse=True)
        X = _dv.fit_transform(bows)

        if norm == "tfidf":
            transformer = TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
            X = transformer.fit_transform(X)
        elif norm == "unit":
            transformer = Normalizer(norm=NORM, copy=False)
            X = transformer.transform(X)
        elif norm == "binary":
            X = binarize(X)
        else:
            pass
            # no normalization

        S = X * X.T
        S = S[:len(bows1),len(bows1):]
        sim_threshold = percentile(S.data, q=q) if len(S.data)> 0 else 1.0
        B = S >= sim_threshold

        if verbose:
            print "senses 1:", len(bows1), len(senses1)
            print "senses 2:", len(bows2), len(senses2)
            print "X:", X.shape
            print "S:", S.shape
            print sim_threshold

        for i, row in enumerate(B):
            for j in row.indices:
                res[senses1[i][0]][senses2[j][0]] = S[i,j]

        return res

