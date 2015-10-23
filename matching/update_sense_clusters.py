from jnt.matching.synset_fetchers import BabelNet, SenseClusters, BABELNET_KEYS
import codecs
from traceback import format_exc
from jnt.matching.synset_fetchers import SEP
from pandas import DataFrame, read_csv


def filter_sc(sc_fpath, output_fpath, voc):
    """ filter sense clusters """

    voc_rows = []

    df = read_csv(sc_fpath, encoding='utf-8', delimiter=SEP, error_bad_lines=False)
    df = df.fillna("")

    for i, row in df.iterrows():
        try:
            if i % 10000 == 0: print i
            if row.word not in voc: continue
            else: voc_rows.append(row)
        except:
            print row
            print format_exc()
    print i + 1

    df = DataFrame(voc_rows, columns=df.columns)
    df.to_csv(output_fpath, sep="\t", encoding="utf-8", float_format='%.6f', index=False)


    print "Output:", output_fpath


def print_sc(sc):
    for i, word in enumerate(sc.data):
        for sid in sc.data[word]:
            print word, sid, sc.data[word][sid]["prob"]


        if i > 100: break


def old2new_ddt(old_fpath, new_fpath):
    ddt_old = SenseClusters(old_fpath, normalized_bow=True)
    voc = ddt_old.data.keys()
    output_fpath = new_fpath + ".out"
    filter_sc(new_fpath, output_fpath, voc)


# old_fpath = "/Users/alex/tmp/matching/ddt-adagram-ukwac+wacky.csv.gz.voc.out"
# new_fpath = "/Users/alex/Desktop/ddt-adagram-ukwac+wacky-464k-closure-v2.csv.gz"
# old2new_ddt(old_fpath, new_fpath)