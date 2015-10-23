import re
from pandas import read_csv
from jnt.common import take
from os.path import splitext, dirname, join
import codecs
from ntpath import basename
from collections import defaultdict, Counter
from jnt.morph import get_stoplist
from plumbum.cmd import zgrep
from jnt.common import load_freq
from jnt.matching.synset_fetchers import SenseClusters
from os.path import join

#re_latin_word = re.compile(ur"^[-a-z]+(#N(N|P))?$", re.U|re.I)
re_latin_word = re.compile(ur"^[-a-z]+(#NN)?$", re.U|re.I)


RELATED_PER_WORD = 500
RELATED_PER_SENSE = 20
MIN_SENSE_FREQ = 1000

def generete_words(voc_fpath, freq_fpath, sc):
    output_all_fpath = join(dirname(voc_fpath), splitext(basename(voc_fpath))[0] + "-clusters.csv")

    freq = load_freq(freq_fpath, min_freq=10, preprocess=True, sep='\t',
              lowercase=False, strip_pos=False, use_pickle=True)

    # load resources
    voc = [r.word for i,r in read_csv(voc_fpath, "\t", encoding='utf8', error_bad_lines=False).iterrows()]
    res = defaultdict(Counter)
    stoplist = get_stoplist()

    # for each model retrieve related words
    for sname in sc:
        output_fpath = join(dirname(voc_fpath), splitext(splitext(basename(sname))[0])[0] + "-clusters.csv")
        with codecs.open(output_fpath, "w", "utf-8") as output:
            print >> output, "word\tnum_related\trelated"
            for w in voc:
                related_cluster = Counter()

                # calculate candidate sense words
                candidates = set()
                for sw in sc[sname].find_word(w):
                    pos = "" if len(sw.split("#")) < 2 else sw.split("#")[1].lower()
                    if pos in ["np","nn",""]: candidates.add(sw)

                print "\n\n\n======================== " + w.upper()

                # for each word candidate
                for wc in candidates:
                    if "#" in wc and freq.get(wc, 0) < MIN_SENSE_FREQ:
                        print "\n\n>>>Skipping:", w, wc, freq.get(wc, 0)
                        continue

                    # for each sense of the candidate
                    print "\n\n>>>", w, wc, freq.get(wc, 0)
                    for sense_id in sorted(sc[sname].data[wc]):

                        # build list of top related words
                        related_words = {}
                        for rw in sorted(sc[sname].data[wc][sense_id]["cluster"], key=sc[sname].data[wc][sense_id]["cluster"].get, reverse=True):
                            if len(related_words) < RELATED_PER_SENSE and re_latin_word.match(rw):
                                rw_lemma = rw.split("#")[0].lower()
                                if rw_lemma not in related_words and rw_lemma not in stoplist:
                                    related_words[rw_lemma] = sc[sname].data[wc][sense_id]["cluster"][rw]

                        related_cluster.update(related_words)
                        #print "+++++++", related_cluster
                        print sense_id,
                        for x in related_words: print x,
                        print ""

                res[w].update(related_cluster)
                related_cluster_s = sorted(related_cluster, key=related_cluster.get, reverse=True)
                print >> output, "%s\t%d\t%s" % (w, len(related_cluster), ','.join(related_cluster_s))
                print ":::%s\t%d\t%s" % (w, len(related_cluster), ','.join(related_cluster_s))


        print "\n\nOutput:", output_fpath

    # Save union of related words for all input datasets
    # related_cluster_words = take(RELATED_PER_WORD, sorted(related_cluster, key=related_cluster.get, reverse=True)
    with codecs.open(output_all_fpath, "w", "utf-8") as output:
        print >> output, "word\tnum_related\trelated"
        for w in sorted(res, key=res.get, reverse=True):
            print >> output, "%s\t%d\t%s" % (w, len(res[w]), ','.join(res[w]))


    print "\n\nOutput:", output_all_fpath




def print_cluster(cluster20):
    for i,related in enumerate(sorted(cluster20, key=cluster20.get, reverse=True)):
        print "\t", i+1, related, cluster20[related]

def load_crowd_clusters(input_fpath):
    return {r.word: r.related.split(",") for i,r in read_csv(input_fpath, "\t", encoding='utf8', error_bad_lines=False).iterrows()}

def run_test(sense_clusters, voc, freq, input_fpath):

    crowd_clusters = load_crowd_clusters(input_fpath)

    for word in voc:
        word_candidates = [word_candidate for word_candidate in sense_clusters.find_word(word)
                           if "#NN" in word_candidate and freq[word_candidate] >= 1000]

        sense_num = 0
        for word_candidate in word_candidates:
            sense_num += len(sc[senses_fpaths[0]].data[word_candidate])

            for senseid in sense_clusters.data[word_candidate]:
                cluster = sense_clusters.data[word_candidate][senseid]["cluster"]
                cluster20 = {}
                print word, word_candidate, senseid, len(cluster)
                for i, related in enumerate(sorted(cluster, key=cluster.get, reverse=True)):
                    if "#NN" in related and len(cluster20) < 20:
                        cluster20[related] = cluster[related]
                        related_lemma = related.split("#")[0].lower()
                        print "\t", i+1, related, cluster20[related], related_lemma in crowd_clusters[word]


# data_dir = "/home/panchenko/joint/data/"
#
# senses_fpaths = ["senses-wiki-n200-380k.csv.gz",
#                  "senses-wiki-n30-1600k.csv.gz",
#                  "senses-news-n200-345k.csv.gz",
#                  "senses-news-n50-485k.csv.gz",
#                  "adagram-200-dt.csv.out"]
# senses_fpaths = map(lambda p: join(data_dir, p), senses_fpaths)
#
# sc = {}
# for fpath in senses_fpaths:
#     sc[fpath] = SenseClusters(fpath, strip_dst_senses=False, load_sim=True, verbose=False)
#
# voc_fpath = join(data_dir, "semeval35n.csv")
# freq_fpath = join(data_dir, "news100M_stanford_cc_word_count.gz")
#
#generete_words(voc_fpath, freq_fpath, sc)
#run_test(sc[senses_fpaths[3]], voc, freq, "/home/panchenko/joint/data/clusters4/semeval35n-clusters.csv")
