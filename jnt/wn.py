from nltk.corpus import wordnet
from traceback import format_exc
import codecs


OFFSET_LEN = 8
VERBOSE = False


def export_wn_lexicon(output_fpath):
    with codecs.open(output_fpath, "w", "utf-8") as output:
        num_lemmas = 0
        for i, synset in enumerate(wordnet.all_synsets()):
            for lemma in synset.lemmas():
                print >> output, lemma.name()
                num_lemmas += 1

        print "Wordnet vocabulary:", output_fpath
        print "# lemmas:", num_lemmas


def sense2offset(word, wn_sense_ids, pos="n"):
    """ convert "python" and "1,2,3") to "wn:09649926,wn:09649923,wn:09649924 """

    try:
        offsets = []
        ids = [int(sense_id) for sense_id in wn_sense_ids.split(",")]
        for i, synset in enumerate(wordnet.synsets(word, pos=pos)):
            sense_id = i + 1
            if sense_id in ids:
                offset = unicode(synset.offset())
                missing_zeros = 8 - len(offset)
                offset = "0" * missing_zeros + offset
                offset = "wn:%sn" % offset
                offsets.append(offset)

        return ','.join(offsets)
    except:
        if VERBOSE:
            print "Error: %s '%s'" % (word, wn_sense_ids)
            print format_exc()
        return ""


class WordNetOffsets(object):
    def __init__(self):
        syns = list(wordnet.all_synsets())
        offsets_list = [(s.offset(), s) for s in syns]
        offsets_dict = dict(offsets_list)
        self._data = offsets_dict

    def get(self, wn_offset):
        """ Given a synset id like 'wn:04743605n' :return a Synset object. """

        if wn_offset in self._data:
            return self._data[wn_offset]
        elif self.wnid_str2int(wn_offset) in self._data:
            return self._data[self.wnid_str2int(wn_offset)]
        else:
            return None

    def wnid_str2int(self, string_id):
        """ wn:08633957n --> 8633957 """

        try:
            return int(string_id[3:-1])
        except:
            return -1