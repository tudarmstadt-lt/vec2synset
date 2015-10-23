from jnt.common import load_voc
import codecs 
from jnt.matching.synset_fetchers import BabelNet, BABELNET_KEYS
from jnt.common import take

MAX_WORDS = 999

voc_fpath = "/Users/alex/work/joint/src/data/ambigous-words-mine.csv"
output_fpath = voc_fpath + "-babelnet.csv"
babelnet_dir = "/Users/alex/tmp/matching/babelnet-eval/"
adagram_voc_fpath = "/Users/alex/tmp/adagram/HugeModel-voc.csv"


babelnet = BabelNet(babelnet_keys=BABELNET_KEYS, babelnet_dir=babelnet_dir,
                    freq_fpath="", divide_by_freq=False)
adagram_voc = load_voc(adagram_voc_fpath)
voc = load_voc(voc_fpath)


with codecs.open(output_fpath, "w", "utf-8") as out:
    for word in voc:
        senses = babelnet.get_senses(word)
        for sense_id, bow in senses:
            bow_words = []
            for w in sorted(bow, key=bow.get, reverse=True):
                if w in adagram_voc and w != word:
                    bow_words.append(w) 
            out.write("%s\t%s\t%s\n" % (word, sense_id, ' '.join(take(MAX_WORDS,bow_words))))
        
print "Output:", output_fpath