from jnt.matching.synset_fetchers import BabelNet, SenseClusters
from jnt.matching.matcher import Matcher

BABELNET_KEY = ""
ADAGRAM_DDT = "/Users/alex/tmp/matching/agagram-hugemodel-ambigous-200-2.csv.out"
voc_fpath = "/Users/alex/work/joint/src/data/ambigous-words.csv"

babelnet = BabelNet(babelnet_key="")
adagram = SenseClusters(ddt_fpath=ADAGRAM_DDT)

m = Matcher(babelnet, adagram)
m.match_file(voc_fpath, voc_fpath + ".match")
