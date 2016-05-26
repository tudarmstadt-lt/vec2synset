from jnt.matching.synset_fetchers import BabelNet, BABELNET_KEYS

babelnet_dir = "/Users/alex/tmp/matching/babelnet/"
freq_fpath = "/Users/alex/tmp/st/word-freq-t10.csv"

babelnet = BabelNet(BABELNET_KEYS, babelnet_dir, freq_fpath=freq_fpath)

