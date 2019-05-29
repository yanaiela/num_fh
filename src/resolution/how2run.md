# Running the NFH models

From this directory, run the following:

```
# Baseline model
allennlp train framework/experiments/base.jsonnet \
        --include-package framework \ 
        -s ../allen_logs/base
      
# Baseline + Elmo model
allennlp train framework/experiments/elmo.jsonnet \
        --include-package framework \
        -s ../allen_logs/elmo
        
# Baseline Oracle with only Reference classes
allennlp train framework/experiments/base_oracle.jsonnet \
        --include-package framework \
        -s ../allen_logs/base_oracle_ref
        
# Baseline Oracle with only Implicit classes
allennlp train framework/experiments/base_oracle.jsonnet \
        --include-package framework \
        -s ../allen_logs/base_oracle_imp \
        --overrides '{"dataset_reader": {"oracle_head": "imp"}}'
        
# Elmo Oracle with only Reference classes
allennlp train framework/experiments/elmo_oracle.jsonnet \
        --include-package framework \
        -s ../allen_logs/elmo_oracle_ref
        
# Elmo Oracle with only Implicit classes
allennlp train framework/experiments/elmo_oracle.jsonnet \
        --include-package framework \
        -s ../allen_logs/elmo_oracle_imp \
        --overrides '{"dataset_reader": {"oracle_head": "imp"}}'
        

```

Then for evaluating the models, with the restrictive evaluation
(which allows not only the closest ref), and a general performance
 breakdown, run the following:

```
python -m src.resolution.framework.evaluate \
        --model src/allen_logs/elmo/model.tar.gz \
        --data_dir data/resolution/processed/ --cuda 0
```
and accordingly with any other relevant model 