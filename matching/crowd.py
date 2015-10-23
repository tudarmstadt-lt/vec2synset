from pprint import pprint

from jnt.patterns import re_whitespaces
import codecs
import xml.etree.ElementTree as et
from glob import glob
from os.path import join
from collections import defaultdict
from jnt.matching.crowdsourcing_words import load_crowd_clusters
from math import ceil
import re

WORDS_PER_PAGE = 5

def clean(text):
    text = text.strip().replace("\n", " ")
    return re_whitespaces.sub(" ", text)


def get_inventory(inventory_dir):
    inventory = defaultdict(dict)

    for word_fname in glob(join(inventory_dir, "*.xml")) :

        tree = et.parse(word_fname)
        root = tree.getroot()

        target = root.attrib["lemma"]
        if "-n" not in target: continue

        for child in root:
            if child.tag == "sense":
                sense = {"target": target}
                sense["id"] = child.attrib["n"]
                sense["name"] = child.attrib["name"]
                for gchild in child:
                    if gchild.tag == "mappings":
                        for ggchild in gchild:
                            if ggchild.tag == "wn": sense["wn"] = ggchild.text
                    if gchild.tag == "commentary":
                        sense["definition"] = clean(gchild.text)
                    if gchild.tag == "examples":
                        sense["example"] = clean(gchild.text)
                inventory[target.replace("-n","")][sense["id"]] = sense
    return inventory


def highlight_target(text, target):
    regex = re.compile(ur"(,|\s|.|^)" + target + "(,|\s|.|$)", re.U|re.I)
    return regex.sub(ur"\1<b>" + target + ur"</b>\2", text)


def build_related4crowd(inventory_dir, related_words_fpath, csv_fpath):

    related = load_crowd_clusters(related_words_fpath)

    with codecs.open(log_fpath, "w", "utf-8") as log, codecs.open(csv_fpath, "w", "utf-8") as table:
        # print header
        print >> table, "id\ttarget\tname\tdefinition\texamples\tontowiki_id\twordnet2_ids",
        for x in range(WORDS_PER_PAGE): table.write("\tmatchterm" + unicode(x+1))
        table.write("\n")

        inventory = get_inventory(inventory_dir)
        for word in inventory:
            if word not in ["president","capital","plant","rate"]: continue
            for sense_id in inventory[word]:
                # print to table
                related_words = list(related[word])
                for chunk in range( int(ceil(float(len(related[word])) / WORDS_PER_PAGE))):
                    table.write("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (word + "#" + sense_id,
                                                 word.strip(),
                                                 inventory[word][sense_id]["name"],
                                                 highlight_target(inventory[word][sense_id]["definition"], word),
                                                 highlight_target(inventory[word][sense_id]["example"], word),
                                                 inventory[word][sense_id]["id"],
                                                 inventory[word][sense_id]["wn"]))
                    for x in range(WORDS_PER_PAGE):
                        try:
                            related_word = related_words.pop()
                        except IndexError:
                            related_word = ""
                        table.write("\t%s" % related_word.strip())
                    table.write("\n")

    print "CSV:", csv_fpath

inventory_dir = "/Users/alex/Desktop/matching-eval/lexical-sample/train/lexical-sample/sense-inventories/"
related_words_fpath = "/Users/alex/Desktop/matching-eval/cluster-terms/semeval35n-clusters.csv"
log_fpath = "/Users/alex/Desktop/matching-eval/related4crowd.txt"
csv_fpath = "/Users/alex/Desktop/matching-eval/related4crowd-tmp.csv"

build_related4crowd(inventory_dir, related_words_fpath, csv_fpath)
