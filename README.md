# Numeric Fused-Head


This is the code used in the paper: 

**"Where’s My Head? Definition, Dataset and Models for Numeric Fused-Heads Identification and Resolution"**
Yanai Elazar and Yoav Goldberg

to appear in TACL

[preprint](https://arxiv.org/pdf/1905.10886.pdf)
[demo](http://nlp.biu.ac.il/~lazary/fh/)


## Installation
Installing the num_fh package is simple as:
```bash
pip install num_fh
```

## Prerequisites
* at least python 3.6 (and specific libraries in the requirements.txt)

We strongly recommend that you use a new virtual environment to install num_fh

### create a clean conda env
```bash
conda create -n nfh python==3.6 anaconda
source activate nfh
```

## Cloning and Running the Library by Yourself
### clone this repo:
```bash
git clone git@github.com:yanaiela/num_fh.git
```

### installing python packages
```bash
pip install -r requirements.txt
```

## Example Usage
```python
import spacy
from num_fh import NFH
nlp = spacy.load('en_core_web_sm')
nfh = NFH(nlp)
nlp.add_pipe(nfh, first=False)

doc = nlp("I told you two, that only one of them is the one who will get 2 or 3 icecreams")
assert doc[16]._.is_nfh == True
assert doc[18]._.is_nfh == False
assert doc[3]._.is_deter_nfh == True
assert doc[16]._.is_deter_nfh == False
assert len(doc._.nfh) == 4
```

The paper (and therefore, also the code) deals with two sub-tasks of the FH solution:
* Identification
* Resolution

These are dealt with separately, and discussed in the paper in sections 
(3,4) and (5,6). These parts are also solved separately in the code, and
contain further instructions for each one in dedicated README files:
[Identification](num_fh/identification/run_files.md) and [Resolution](num_fh/resolution/how2run.md)

## Citing
If you find this work relevant to yours, please consider citing us:
```
@misc{elazar2019wheres,
    title={Where's My Head? Definition, Dataset and Models for Numeric Fused-Heads Identification and Resolution},
    author={Yanai Elazar and Yoav Goldberg},
    year={2019}
}
```

## Contact
If you have any question, issue or suggestions, feel free to contact 
us with the emails listed in the paper.
