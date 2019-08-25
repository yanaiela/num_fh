// Configuration for a the baseline model with w2v embeddings used in tacl paper.
// This version uses oracle separation between Implicit and Reference

local data_path = "../../data/resolution/processed/";
local emb_dim = 300;
local emb_out = 200;
local char_dim = 30;
local char_out = 50;
{
  "dataset_reader": {
    "type": "nfh_orcale_reader",
    "token_indexers": {
            "tokens": { "type": "single_id" },
            "token_characters": { "type": "characters" }
        },
    "oracle_head": "ref"
  },
  "train_data_path": "https://storage.googleapis.com/ai2i/datasets/num_fh/nfh_train.jsonl",
  "validation_data_path": "https://storage.googleapis.com/ai2i/datasets/num_fh/nfh_dev.jsonl",
  "test_data_path": "https://storage.googleapis.com/ai2i/datasets/num_fh/nfh_test.jsonl",
  "evaluate_on_test": true,

  "model": {
    "type": "nfh_model_base",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/word2vec/GoogleNews-vectors-negative300.txt.gz",
            "embedding_dim": emb_dim,
            "trainable": false
        },
        "token_characters": {
             "type": "character_encoding",
             "embedding": {
                 "embedding_dim": char_dim,
             },
             "encoder": {
                 "type": "lstm",
                 "input_size": char_dim,
                 "hidden_size": char_out,
                 "dropout": 0.2
             },
         }
      }
    },
    "encoder": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": emb_dim + char_out,
        "hidden_size": emb_out,
        "num_layers": 1,
        "dropout": 0.2
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["sentence", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 20,
    "grad_norm": 1.0,
    "patience" : 5,
    "cuda_device" : 1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  }
}
