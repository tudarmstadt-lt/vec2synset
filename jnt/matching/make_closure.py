from pandas import read_csv
import codecs
from jnt.common import preprocess_pandas_csv

def remove_unknown(cluster_str):
    try:
        return ','.join([entry for entry in cluster_str.split(",") if "?" not in entry])
    except:
        print "Error:", cluster_str
        return ""
    
def process(ddt_fpath):

    preprocess_pandas_csv(ddt_fpath)
    df = read_csv(ddt_fpath, encoding='utf-8', delimiter="\t", error_bad_lines=False)
    df = df.fillna("")
    print len(df), "senses loaded"

    closure_fpath = ddt_fpath + ".closure"
    with codecs.open(closure_fpath, "w", "utf-8") as closure:
        print >> closure, "word\tcid\tcluster\tisas"
        for i, row in df.iterrows():
            cluster = remove_unknown(row.cluster)
            isas = remove_unknown(row.isas)
            print >> closure, "%s\t%s\t%s\t%s" % (row.word, row.cid, cluster, isas)

    print "Output:", closure_fpath
    
    
# for fpath in ["/home/panchenko/joint/data/closure/ddt-news-n200-345k.csv",
#               "/home/panchenko/joint/data/closure/ddt-news-n50-485k.csv",
#               "/home/panchenko/joint/data/closure/ddt-wiki-n200-380k-v3.csv"]:
#     process(fpath)


for fpath in ["/Users/alex/tmp/ddt-wiki-n30-1400k-v3.csv",]:
    process(fpath)
    # delete manually lines with ? -- they are broken!
