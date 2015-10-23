import requests
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()
import ujson as json
import re
from jnt.morph import tokenize, lemmatize_word
from os.path import splitext
from time import time
from pandas import read_csv
import codecs
from collections import defaultdict, Counter
from traceback import format_exc
import cPickle as pickle
from jnt.common import exists
import operator
from jnt.common import ensure_dir
from os.path import join
from jnt.morph import get_stoplist
from jnt.patterns import re_spaced_numbers
from time import sleep
from jnt.common import load_voc
from jnt.utils.freq import load_freq
import gzip
import codecs
from jnt.utils.freq import FreqDictionary

BABELNET_KEYS = [""]
BABELNET_ENDPOINT = "https://babelnet.io/v1/"
BABELNET_SLEEP = 0 # seconds
HEADERS = {"Accept-Encoding": "gzip"}
GET_SYNSET_IDS = "getSynsetIds"
GET_SYNSET = "getSynset"
DEFAULT_LANG = "EN"
GET = True
CODE_OK = 200
IMAGE_STOP_LIST = ["logo"]
LIMIT_MSG  = {u'message': u'Your key is not valid or the daily requests limit has been reached. Please visit http://babelnet.org.'}
V = True # Verbose
SEP = "\t"
SEP2 = ","
SEP4 = "#"
SEP3 = ":"
STRIP_DST_SENSES = True
BABELNET_PKL = "babelnet.pkl"

# Meta-parameters of the method
REMOVE_BOW_STOPWORDS = True
LEMMATIZE_BOW = True
LOWERCASE_BOW = True


_re_norm_babel = re.compile(ur"[()_:]", re.U|re.I)
_re_norm_babel_dash = re.compile(ur"[()_:-]", re.U|re.I)
_re_whitespaces2 = re.compile(r"\s+")
_stoplist = get_stoplist()


def good_token(w):
    return (w not in _stoplist and
            not re_spaced_numbers.match(w))


class DailyLimitException(Exception):
    def __init__(self):
        pass


class BabelNet(object):
    def __init__(self, babelnet_keys, babelnet_dir="", freq_fpath="", normalized=True, divide_by_freq=False, force_api=False):
        self._babelnet_keys = babelnet_keys
        self._babelnet_dir = babelnet_dir
        ensure_dir(self._babelnet_dir)
        self._normalized = normalized
        self._force_api = force_api
        self._freq =  FreqDictionary(freq_fpath)
        self._babelnet = self._load(babelnet_dir, divide_by_freq=divide_by_freq) # (word->sense_id->{"bow" , "wnOffset"}

    @property
    def data(self):
        return self._babelnet

    def _load(self, babelnet_dir, divide_by_freq=False, sanity_check=True):
        babelnet_fpath = join(babelnet_dir, BABELNET_PKL)
        if not exists(babelnet_fpath): return defaultdict(dict)

        with open(babelnet_fpath, 'rb') as babelnet_file:
            bn = pickle.load(babelnet_file)

        if sanity_check:
            err_num = 0
            for word in bn:
                if len(bn[word]) <= 0:
                    err_num += 1
                    print "Warning: local word with no senses", word
            if err_num > 0:
                print "Warning:", err_num, "local words with no senses"

            print "Loaded BabelNet with %d words from: %s" % (len(bn), babelnet_fpath)

        self._block_save = False
        if self._normalized:
            for word in bn:
                for sense_id in bn[word]:
                    if divide_by_freq:
                        bow = Counter({w: bn[word][sense_id]["bow"][w] / self._freq.freq(w) for w in bn[word][sense_id]["bow"] if good_token(w)})
                        self._block_save = True
                    else:
                        bow = bn[word][sense_id]["bow"]

                    max_freq_norm = float(max(bow.values())) if len(bow) > 0 else 1.0
                    if max_freq_norm == 0.0: max_freq_norm = 1.0
                    bow_range_norm = Counter({w: bow[w] / max_freq_norm for w in bow if good_token(w)})

                    bn[word][sense_id]["bow"] = bow_range_norm


        return bn

    def wn_mapping(self):
        words = self.data.keys()
        wn_ids = defaultdict(dict)
        for word in words:
            for bn_id in self.data[word]:
                if len(self.data[word][bn_id]["wnOffsets"]) == 0: continue
                for wnid_dict in self.data[word][bn_id]["wnOffsets"]:
                    wn_ids[word][bn_id] = []
                    if "id" not in wnid_dict: continue
                    else: wn_ids[word][bn_id].append(wnid_dict["id"])

                if len(wn_ids[word][bn_id]) > 1:
                    print "Warning: more than two mappings", word, bn_id, wn_ids[word][bn_id]

        return wn_ids

    def save(self, babelnet_dir=""):
        if self._block_save:
            print "Save blocked"
            return

        if babelnet_dir == "" and self._babelnet_dir == "":
            print "Error: specify path to the output file"
            return
        babelnet_fpath = join(babelnet_dir, BABELNET_PKL) if babelnet_dir != "" else join(self._babelnet_dir, BABELNET_PKL)
        with open(babelnet_fpath, 'wb') as babelnet_file:
            pickle.dump(self._babelnet, babelnet_file)

        print "BabelNet saved to:", babelnet_fpath

    def _get_key(self):
        return self._babelnet_keys[0]

    def _get_synset_ids(self, word, lang=DEFAULT_LANG, pos=""):
        params = {"word": word, "lang":lang.upper(), "key": self._get_key()}
        response = requests.get(BABELNET_ENDPOINT + GET_SYNSET_IDS, params=params, headers=HEADERS, verify=True)

        if response.status_code == CODE_OK:
            content = json.loads(response.content)
            if content == LIMIT_MSG and len(self._babelnet_keys) > 1:
                self._babelnet_keys.pop(0)
                return self._get_synset_ids(word, lang=lang, pos=pos)
            elif content == LIMIT_MSG:
                raise DailyLimitException()
            else:
                return map(lambda x: x["id"], content)
        else:
            print "Error: cannot process query '%s'. Status code: %d.\n" % (word, response.status_code)
            return []

    def _get_synset(self, synset_id):
        params = {"id": synset_id, "key": self._get_key()}
        response = requests.get(BABELNET_ENDPOINT + GET_SYNSET, params=params, headers=HEADERS, verify=True)

        if response.status_code == CODE_OK:
            content = json.loads(response.content)
            if content == LIMIT_MSG:
                print "Error: BabelNet daily limit is over."
                return {}
            else:
                return content
        else:
            print "Error: cannot process query '%s'. Status code: %d.\n" % (word, response.status_code)
            return {}

    def get_wordnet_senseids(self, word):
        """ Returns a list of dicts {'wordnet': sense_id, 'babelnet': sense_id} """

        senses = []
        if word not in self._babelnet: return senses

        for babelnet_id in self._babelnet[word]:
            if "wnOffsets" not in self._babelnet[word][babelnet_id]:
                print "Warning:", babelnet_id, "no wnOffsets"
                continue

            if len(self._babelnet[word][babelnet_id]["wnOffsets"]) == 0:
                print "Warning:", babelnet_id, "no wordnet senses"
                continue

            for wn_sense in self._babelnet[word][babelnet_id]["wnOffsets"]:
                if "id" not in wn_sense:
                    print "Warning:", babelnet_id, "no id"
                    continue
                senses.append({'babelnet': babelnet_id, 'wordnet': wn_sense["id"]})

        return senses


    def _normalize(self, word, dash=False):
        word = _re_norm_babel_dash.sub(u" ", word) if dash else _re_norm_babel.sub(u" ", word)
        word = _re_whitespaces2.sub(u" ", word)
        return word.lower().strip()

    def _get_synset_bow(self, synset, lang=DEFAULT_LANG, senses=True, glosses=False, categories=False, image_names=False):
        bow = Counter()
        if senses and "senses" in synset:
            for s in synset["senses"]:
                if s["language"] != lang: continue
                lemma = self._normalize(s["lemma"])
                bow[lemma] += 1
                bow.update(lemma.split(" "))
                slemma = self._normalize(s["simpleLemma"])
                bow[slemma] += 1
                bow.update(slemma.split(" "))

        if glosses and "glosses" in synset:
            for s in synset["glosses"]:
                if s["language"] != lang: continue
                bow.update(tokenize(s["gloss"], lowercase=LOWERCASE_BOW, remove_stopwords=REMOVE_BOW_STOPWORDS))

        if categories and "categories" in synset:
            for s in synset["categories"]:
                if s["language"] != lang: continue
                bow.update(tokenize(self._normalize(s["category"]), lowercase=LOWERCASE_BOW, remove_stopwords=REMOVE_BOW_STOPWORDS))

        if image_names and "images" in synset:
            names = set(s["name"] for s in synset["images"] if s["name"])
            map(lambda n: bow.update(
                filter(lambda t: t not in IMAGE_STOP_LIST, tokenize(self._normalize(splitext(n)[0], dash=True)))
                ), names)

        return bow

    def fetch_from_voc(self, voc_fpath):
        for i, row in read_csv(voc_fpath, "\t", encoding='utf8', error_bad_lines=False).iterrows():
            try:
                s = self.get_senses(row.word)  # saving to self._babelnet
                print row.word, len(s) 
            except KeyboardInterrupt:
                self.save()
                return
            except DailyLimitException:
                print "Error: Daily limit exceeded"
                self.save()
                return
            except:
                print "Error:", row
                print format_exc()
                break
        self.save()


    def _save_synset(self, word, sid, synset):
        try:
            if not exists(self._babelnet_dir): return
            output_fpath = join(self._babelnet_dir, word + "#" + sid + ".json")
            with codecs.open(output_fpath, 'w', "utf-8") as outfile:
                print >> outfile, json.dumps(synset, ensure_ascii=False).decode("utf-8")
        except:
            print "Error saving file"
            print format_exc()

    def get_senses(self, word, lang=DEFAULT_LANG, senses=True, glosses=True, categories=True, image_names=True,
                   verbose=False, min_prob=0.0):
        """ Returns a list of tuples (sense_id, bow), where bow is a Counter and sense_id is a unicode """

        if word in self._babelnet and not self._force_api:
            senses_lst = [(sid, self._babelnet[word][sid]["bow"]) for sid in self._babelnet[word]]
            if verbose: print  word, ": local"
        else:
            senses_lst = []
            for i, sid in enumerate(self._get_synset_ids(word, lang)):
                tic = time()
                synset = self._get_synset(synset_id=sid)
                if len(synset) == 0: continue
                bow = self._get_synset_bow(synset, senses=senses, glosses=glosses, categories=categories, image_names=image_names)
                senses_lst.append((sid, bow))
                print "%s#%d\t%s\t%.2f sec." % (word, i, sid, time()-tic)

                self._babelnet[word][sid] = {"bow": bow, "wnOffsets": synset.get("wnOffsets", "")}
                self._save_synset(word, sid, synset)

                if verbose:
                    print "\n%s#%d\t%s\n===================================================" % (word.upper(), i, sid)
                    print "senses=True, glosses=True, categories=True, image_names=True"
                    print self._get_synset_bow(synset, senses=True, glosses=True, categories=True, image_names=True)

                    print "\nsenses=True"
                    print self._get_synset_bow(synset, senses=True, glosses=False, categories=False, image_names=False)

                    print "\nglosses=True"
                    print self._get_synset_bow(synset, senses=False, glosses=True, categories=False, image_names=False)

                    print "\ncategories=True"
                    print self._get_synset_bow(synset, senses=False, glosses=False, categories=True, image_names=False)

                    print "\nimage_names=True"
                    print self._get_synset_bow(synset, senses=False, glosses=False, categories=False, image_names=True)

            print word, ": api"
            sleep(BABELNET_SLEEP)

        return senses_lst

    def get_cluster(self, word, sense_id):
        if word in self._babelnet and sense_id in self._babelnet[word]:
            return self._babelnet[word][sense_id]["bow"]
        else: return []


def filter_ddt_by_voc(ddt_fpath, voc_fpath, ddt_filtered_fpath):

    # ddt_fpath = "/Users/alex/tmp/matching/ddt-adagram-ukwac+wacky.csv.gz"
    # voc_fpath = "/Users/alex/work/joint/src/data/ambigous-words.csv"
    # ddt_filtered_fpath = ddt_fpath + ".voc"

    voc = load_voc(voc_fpath)

    with codecs.open(ddt_filtered_fpath, "w", "utf-8") as out:
        num = 0
        found_voc = set()
        for i, line in enumerate(gzip.open(ddt_fpath, "rb", "utf-8")):
            if i % 100000 == 0: print i, num
            f = line.split("\t")
            if len(f) < 1: continue
            if f[0] in voc:
                num += 1
                found_voc.add(f[0])
                out.write(line)

    print "Input processed:", i
    print "Words found:", len(found_voc), "of", len(voc)
    print "Senses written:", num
    print "Filtered by vocabulary DDT:", ddt_filtered_fpath


def trim_sense_clusters(voc_fpath):
    pass


class SenseClusters(object):
    def __init__(self, sense_clusters_fpath, strip_dst_senses=False, load_sim=True, verbose=V, normalized_bow=False):
        """ Loads and operates sense clusters in the format 'word<TAB>cid<TAB>prob<TAB>cluster<TAB>isas' """

        self._verbose = verbose
        self._normalized_bow = normalized_bow
        self._sc, self._normword2word = self._load(sense_clusters_fpath, strip_dst_senses, load_sim)

    @property
    def words(self):
        return self._sc.keys()

    @property
    def normwords(self):
        return self._normword2word.keys()

    @property
    def data(self):
        return self._sc

    def find_word(self, word):
        return self._normword2word.get(self.norm(word), "")

    def _get_words(self, words_str, strip_dst_senses, load_sim):
        cluster_words = Counter()
        if words_str == "": return cluster_words 
        for j, cw in enumerate(words_str.split(SEP2)):
            try:
                fields = cw.split(SEP3)
                if j == 0:
                    max_sim = float(fields[1]) if len(fields) > 1 else 1.0
                word = fields[0].strip()
                if load_sim:
                    sim = float(fields[1]) if len(fields) >= 2 else 1.0/(j+1.0)**0.33
                    if sim > 1.0: sim = sim / max_sim
                else:
                    sim = 1.0

                if strip_dst_senses: word = word.split(SEP4)[0]

                if not self._normalized_bow or good_token(word):
                    cluster_words[word] = float(sim)
            except:
                if self._verbose:
                    print "Warning: bad word", cw
                    print format_exc()
        return cluster_words

    def _get_normalized_words(self, cluster_words):
        res = {}
        for w in cluster_words:
            token = self.norm(w)
            lemma = lemmatize_word(token) if LEMMATIZE_BOW else token
            if REMOVE_BOW_STOPWORDS and token in _stoplist or lemma in _stoplist: continue

            res[lemma] = cluster_words[w]
            res[token] = cluster_words[w]

        return res

    def norm(self, word):
        if LOWERCASE_BOW: return word.split(SEP4)[0].lower()
        else: return word.split(SEP4)[0]

    def _load(self, ddt_fpath, strip_dst_senses, load_sim):
        """ Loads a dict[word][sense] --> {"cluster": Counter(), "cluster_norm": Counter(), "isas": Counter()} """

        df = read_csv(ddt_fpath, encoding='utf-8', delimiter=SEP, error_bad_lines=False)
        df = df.fillna("")
        senses = defaultdict(dict)
        normword2word = defaultdict(set)

        err_clusters = 0
        num_senses = 0
        i = 0

        # foreach sense cluster
        for i, row in df.iterrows():
            try:
                if i % 100000 == 0: print i
                r = {}
                r["prob"] = row.prob if "prob" in row else 1.0
                r["cluster"] = self._get_words(row.cluster, strip_dst_senses, load_sim) if "cluster" in row else Counter()
                r["cluster_norm"] = self._get_normalized_words(r["cluster"])
                r["isas"] = self._get_words(row.isas, strip_dst_senses, load_sim) if "isas" in row else Counter()
                r["isas_norm"] =  self._get_normalized_words(r["isas"])
                senses[row.word][row.cid] = r
                normword2word[self.norm(row.word)].add(row.word)
                num_senses += 1
            except:
                if self._verbose:
                    print "Warning: bad cluster"
                    print row
                    print format_exc()
                err_clusters += 1
        
        print err_clusters, "cluster errors"
        print num_senses, "senses loaded out of", i + 1
        print len(senses), "words loaded"

        return senses, normword2word


    def _normalize(self, word, dash=False):
        word = _re_norm_babel_dash.sub(u" ", word) if dash else _re_norm_babel.sub(u" ", word)
        word = _re_whitespaces2.sub(u" ", word)
        return word.lower().strip()

    def _filter_cluster(self, cluster):
        return Counter({w: cluster[w] for w in cluster if good_token(w)})

    def get_senses(self, word, verbose=False, min_prob=0.0):
        """ Returns a list of tuples (sense_id, bow), where bow is a Counter and sense_id is a unicode """

        if word not in self._sc:
            return []
        else:
            field = "cluster_norm" if self._normalized_bow else "cluster"
            return [(unicode(cid), self._filter_cluster(self._sc[word][cid][field]))
                    for cid in self._sc[word] if self._sc[word][cid]["prob"] > min_prob]

    def get_cluster(self, word, sense_id):
        """ Returns cluster of a given word sense  """

        field = "cluster_norm" if self._normalized_bow else "cluster"

        if word in self._sc and sense_id in self._sc[word]:
            return self._sc[word][sense_id][field]
        elif word in self._sc and unicode(sense_id) in self._sc[word]:
            return self._sc[word][unicode(sense_id)][field]
        elif word in self._sc:
            try:
                return self._sc[word][int(sense_id)][field]
            except:
                return Counter()
        else: return Counter()
