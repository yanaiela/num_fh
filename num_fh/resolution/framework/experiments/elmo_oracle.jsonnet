// Configuration for a the baseline model with Elmo embeddings used in tacl paper.
// This version uses oracle separation between Implicit and Reference

local data_path = "../../data/resolution/processed/";
local emb_dim = 1024;
local emb_out = 200;
{
  "dataset_reader": {
    "type": "nfh_orcale_reader",
    "token_indexers": {
			"elmo": {
		    	"type": "elmo_characters"
    		}
	},
    "oracle_head": "ref"
  },
  "train_data_path": "https://storage.googleapis.com/ai2i/datasets/num_fh/nfh_train.jsonl",
  "validation_data_path": "https://storage.googleapis.com/ai2i/datasets/num_fh/nfh_dev.jsonl",
  "test_data_path": "https://storage.googleapis.com/ai2i/datasets/num_fh/nfh_train.jsonl",
  "evaluate_on_test": true,

  "model": {
    "type": "nfh_model_base",
    "text_field_embedder": {
      "token_embedders": {
			"elmo": {
   	 			"type": "elmo_token_embedder",
    			"options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    			"weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
    			"do_layer_norm": false,
    			"dropout": 0.5
  			}
      }
    },
    "encoder": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": emb_dim,
        "hidden_size": emb_out,
        "num_layers": 1,
        "dropout": 0.2
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["sentence", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 16
  },
  "trainer": {
    "num_epochs": 20,
    "grad_norm": 1.0,
    "patience" : 5,
    "cuda_device" : 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
    }
  },
}
