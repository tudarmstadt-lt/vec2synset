from jnt.matching.synset_fetchers import BabelNet, SenseClusters
from jnt.matching.matcher import Matcher

BABELNET_KEY = ""   # insert a BabelNet API key here (not required)
ADAGRAM_BOW = "data/ddt-adagram-50.csv"  # bow with Adagram nearest neighbours
BABELNET_BOW = "data/babelnet-bow-5190.pkl"  # wget http://panchenko.me/data/joint/adagram/data/babelnet-bow-5190.pkl
VOC = "data/voc-50.csv"  # the mapping will be performed for these words
OUTPUT = VOC + ".match.csv"

babelnet = BabelNet(babelnet_keys=[BABELNET_KEY], babelnet_fpath=BABELNET_BOW)
adagram = SenseClusters(sense_clusters_fpath=ADAGRAM_BOW, strip_dst_senses=True)
m = Matcher(babelnet, adagram)
m.match_file(words_fpath=VOC, output_fpath=OUTPUT)

