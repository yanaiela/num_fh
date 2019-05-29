# NFH Identification

This script is used for finding numeric fused-heads over large corpora using
constituency trees.
It makes use of an imdb corpora, and a python wrapper over stanfordnlp
with using the heuristic of finding an NP phrase with no noun (or equivalent).
The process described here is for an efficient and parallel processing of the
raw data, the parsing into trees, automatically labeling the sentences as
FH or not-FH, and finally creating an ML model for solving it.
 
### Converting a corpora into sentences
```
mkdir data/identification/imdb
python src/identification/data/process_sentences.py \
  --imdb_file data/identification/imdb/shows.json \
  --out_file data/identification/imdb/trees/sentences.txt
```
    

### Parsing the sentences
```
cd data/identification/imdb/trees/

# Split large file into several small ones:
split -l 100000 -d sentences.txt sentences_
# Write into a file all the sub-files
ls -d -1 $PWD/sentences_* > sentences_
# Run Stanford parser on all files:
cd ../../../../
java -Xmx200g -cp "$STANFORD_CORENLP_HOME/*" edu.stanford.nlp.pipeline.StanfordCoreNLP \
  -annotators tokenize,ssplit,parse \
  -filelist data/identification/imdb/trees/sentences_ \
  -outputFormat text \
  -parse.nthreads 40 \
  -ssplit.eolonly true

```

### Detecting FH over the parsed tree
```
# update the script find_fh.py with the parameter stanford_parser_dir with the
# relevant directory of stanford parser
python src/identification/data/tree2numeric_fh.py \
  --imdb_file data/identification/imdb/shows.json \
  --trees_path data/identification/imdb/trees/ \
  --out_path data/identification/processed/
```    

### Building features
```
# Building and saving the data features in a pickle for 
# fast experiments with models
python src/identification/features/build_features.py \
  --data_dir data/identification/processed/ \
  --out_dir data/identification/pickled/ \
  --window_size 3
```

### Running the model
```
python src/identification/models/linear_model.py \
  --data_dir data/identification/pickled/ \
  --model_out models/is_nfh.pkl 
```
