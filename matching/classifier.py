import xml.etree.ElementTree as et
from jnt.common import load_voc
from jnt.wn import sense2offset
import codecs
from pandas import read_csv, merge, Series
import argparse
from os.path import splitext
from os.path import join
from jnt.common import exists
from subprocess import Popen, PIPE
import os
from os.path import splitext
from jnt.morph import get_stoplist
from jnt.patterns import re_number


ADAGRAM_VOC = "/Users/alex/tmp/adagram/HugeModel-voc.csv"
DEFAULT_MAPPING = "/Users/alex/work/joint/src/data/best-matching-out.csv"
DYLD_LIBRARY = "/Users/alex/tmp/adagram/AdaGram.jl/lib/"
ADAGRAM_SCRIPTS_DIR = "/Users/alex/work/joint/src/jnt/adagram/"

_adagram_voc = load_voc(ADAGRAM_VOC, silent=True)
_stoplist = get_stoplist()


def filter_voc(text):
    text_adagram = [w.lower() for w in text.split(" ") if w in _adagram_voc]
    return " ".join(text_adagram)


TARGET_BEG = "((("
TARGET_END = ")))"


def filter_context(context, target, remove_target, context_size):
    context = [w for w in context.split(" ") if w.strip() != "" and w not in _stoplist and not re_number.match(w)]
    if remove_target:
        context = [w for w in context if w != target]
    context = list(set(context))
    context = ' '.join(context[-context_size:])
    return context


def get_context(context, remove_target, context_size):
    x = context.split(TARGET_BEG)
    if len(x) == 2:
        left = x[0]
        y = x[1].split(TARGET_END)
        if len(y) == 2:
            target = y[0].strip()
            right = y[1]

            left = filter_context(left, target, remove_target, context_size)
            right = filter_context(right, target, remove_target, context_size)
            res = left + " " + right

            return res

        else:
            return context
    else:
        return context


def semeval_xml2csv(train_fpath, output_fpath, remove_target=True, context_size=100):
    tree = et.parse(train_fpath)
    root = tree.getroot()

    with codecs.open(output_fpath, "w", "utf-8") as out:
        for child in root:
            if child.tag == "lexelt":
                if child.attrib["pos"] != "n": continue
                word = child.attrib["item"][:-2]
                for gchild in child:
                    if gchild.tag != "instance": continue
                    context = {"word": word}
                    for ggchild in gchild:
                        if ggchild.tag == "context":
                            context["context"] = filter_voc(get_context(ggchild.text, remove_target, context_size))
                        elif ggchild.tag == "answer":
                            context["wn_ids"] = sense2offset(word, ggchild.attrib["wn"]).strip()

                    if len(context["wn_ids"]) == 0: continue
                    out.write("%(word)s\t%(wn_ids)s\t%(context)s\n" % context)

        print "Output:", output_fpath


def evaluate_disambiguated(mapping_fpath, disambiguated_fpath, output_fpath):

    # Merge predictions and golden standard data
    mapping_df = read_csv(mapping_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    disambiguated_df = read_csv(disambiguated_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    res_df = merge(disambiguated_df, mapping_df, how='inner', on=["word","adagram_id"])

    # Calculate performance metrics
    res_df = res_df.fillna("")
    res_df["gold_wn_match"] = Series("", res_df.index)
    res_df["gold_bn_match"] = Series("", res_df.index)

    for i, row in res_df.iterrows():
        golden_ids = row.golden_id.split(",")
        res_df.loc[i, "gold_wn_match"] = row.wordnet_id in golden_ids
        res_df.loc[i, "gold_bn_match"] = row.babelnet_id in golden_ids

    print "# input texts:", len(disambiguated_df)
    print "# babelnet mappings: %d, %.2f%%" % ((i+1), 100*(float(i+1)/ len(disambiguated_df)))
    print "Accuracy (wordnet all babelnet): %.3f" % (float(sum(res_df.gold_wn_match)) / (i+1))
    print "# wordnet mappings:  %d, %.2f%%" % (sum(res_df.wordnet_id != ""), 100.* sum(res_df.wordnet_id != "") / len(disambiguated_df))
    print "Accuracy (wordnet): %.3f, %d" % (float(sum(res_df.gold_wn_match))/sum(res_df.wordnet_id != ""), sum(res_df.gold_wn_match))
    print "Accuracy (babelnet): %.3f, %d" % (float(sum(res_df.gold_bn_match))/sum(res_df.babelnet_id != ""), sum(res_df.gold_bn_match))
    print sum(res_df.golden_id == res_df.babelnet_id), len(res_df)

    # Save results
    res_df.to_csv(output_fpath, sep="\t", encoding="utf-8", float_format='%.3f', index=False)
    print "Output:", output_fpath

    return res_df


def groupby_evaluation(res_df, output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as out:
        out.write("word\tgolden_id\tadagram_id\tcontext\tadagram_prob\tbabelnet_id\twordnet_id\tbabelnet_match\twordnet_match\n")

        babelnet_match_num = 0.
        wordnet_match_num = 0.
        text_num = 0.

        for key, rows in res_df.groupby(["word","golden_id","adagram_id","context","adagram_prob"]):
            text_num += 1
            babelnet_ids = set()
            wordnet_ids = set()
            for i, row in rows.iterrows():
                if row.babelnet_id != "": babelnet_ids.add(row.babelnet_id)
                if row.wordnet_id != "": wordnet_ids.add(row.wordnet_id)

            golden_ids = set(key[1].split(","))
            babelnet_match = int(len(golden_ids.intersection(babelnet_ids)) > 0)
            if babelnet_match: babelnet_match_num += 1
            wordnet_match = int(len(golden_ids.intersection(wordnet_ids)) > 0)
            if wordnet_match: wordnet_match_num += 1
            if len(wordnet_ids) == 0:
                continue
            out.write("%s\t%s\t%s\t%s\t%.3f\t%s\t%s\t%d\t%d\n" % (key[0], ",".join(golden_ids), key[2], key[3], key[4],
                ",".join(babelnet_ids), ",".join(wordnet_ids), babelnet_match, wordnet_match))

        print "Accuracy (babelnet): %.2f" % (babelnet_match_num/text_num)
        print "Accuracy (wordnet): %.2f" % (wordnet_match_num/text_num)
        print "Output:", output_fpath


def adagram_disambiguate(contexts_fpath, model_fpath, output_fpath, nearest_neighbors="false"):
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = DYLD_LIBRARY
    p = Popen(["julia",
               join(ADAGRAM_SCRIPTS_DIR, "matching.jl"),
               contexts_fpath,
               model_fpath,
               output_fpath,
               nearest_neighbors],
               stdin=PIPE,
               stdout=PIPE,
               stderr=PIPE,
               env=env)
    stdout, err = p.communicate(b"")
    rc = p.returncode

    print stdout
    print err
    print "Output:", output_fpath
    print "Output exits:", exists(output_fpath)


def classify(contexts_fpath, model_fpath, mapping_fpath, output_fpath=""):
    """ Performs WSD of the contexts provided in the format 'word<TAB>sense_id<TAB>context' """

    base_name = splitext(contexts_fpath)[0] if output_fpath == "" else output_fpath
    ag_fpath = base_name + "-ag.csv"
    adagram_disambiguate(contexts_fpath, model_fpath, ag_fpath)
    print "Disambiguated:", ag_fpath

    ag_wn_bn_fpath = base_name + "-ag-bn-wn.csv"
    res_df = evaluate_disambiguated(mapping_fpath, ag_fpath, ag_wn_bn_fpath)
    print "Disambiguated with mappings:", ag_wn_bn_fpath

    ag_wn_bn_group_fpath = base_name + "-ag-bn-wn-group.csv"
    groupby_evaluation(res_df, ag_wn_bn_group_fpath)

    print "Disambiguated with mapping, grouped:", ag_wn_bn_group_fpath

    return ag_wn_bn_group_fpath

def main():
    parser = argparse.ArgumentParser(description='Perform disambiguation with BabelNet/WordNet sense labels.')
    parser.add_argument('input', help='Path to a file with input file "word<TAB>golden-sense-ids<TAB>context".')
    parser.add_argument('-o', help='Output file. Default -- next to input file.', default="")
    args = parser.parse_args()

    output_fpath = splitext(args.input)[0] + "-disambiguated.csv" if args.o == "" else args.o
    print "Input: ", args.input
    print "Output: ", output_fpath
    print "Mapping:", DEFAULT_MAPPING
    classify(args.input, DEFAULT_MAPPING, output_fpath)

if __name__ == '__main__':
    main()
