# vec2synset

A tool for matching word sense embeddings with synsets of lexical resources: the method helps to make word sense embeddings interpretable. 

See the website for description of the method: http://tudarmstadt-lt.github.io/vec2synset

### Quickstart 

1. Clone repository: 

  ```
  git clone https://github.com/tudarmstadt-lt/vec2synset.git
  ```

2. Install dependencies:

  ```
  pip install -r requirements.txt
  ```

3. Download resources into the repository (100 Mb):

  ```
  cd vec2synset/data && wget http://panchenko.me/data/joint/adagram/babelnet-bow-5190.pkl
  ```
 
4. Run a test matching (50 words):
 
```
python run.py
```
 
You should get the file with matched AdaGram and BabelNet senses similar to the one in ```data/voc-50.csv.match_example.csv```.

