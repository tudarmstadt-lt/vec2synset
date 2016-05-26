from jnt.matching.classifier import semeval_xml2csv

train_fpath = "/Users/alex/tmp/matching/evaluation/wsd/train.xml"
train_fpath = "/Users/alex/tmp/matching/evaluation/wsd/english-lexical-sample.train.xml"
output_fpath = train_fpath + "-out.csv"

semeval_xml2csv(train_fpath, output_fpath)