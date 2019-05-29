# Numeric Fused-Head


This is the code used in the paper: 

**"Where’s My Head? Definition, Dataset and Models for Numeric Fused-Heads Identification and Resolution"**
Yanai Elazar and Yoav Goldberg,
to appear in TACL
[preprint](https://arxiv.org/pdf/1905.10886.pdf)
and there's also a [demo](http://nlp.biu.ac.il/~lazary/fh/)!


### Prerequisites
* python 3.6 (and specific libraries in the requirements.txt)

```sh
# clone this repo:
git clone git@github.com:yanaiela/num_fh.git

# create a clean conda env
conda create -n nfh python==3.6 anaconda
source activate nfh

# additional python pacakges from requirements.txt
pip install -r requirements.txt

```

The paper deals with two sub-tasks of the FH solution:
* Identification
* Resolution

These are dealt with separately, and discussed in the paper in sections 
(3,4) and (5,6). These parts are also solved separately in the code, and
contain further instructions for each one in a dedicated README file.

### Citing
If you find this work relevant to yours, please consider citing us:
```
@misc{elazar2019wheres,
    title={Where's My Head? Definition, Dataset and Models for Numeric Fused-Heads Identification and Resolution},
    author={Yanai Elazar and Yoav Goldberg},
    year={2019}
}
```

### Contact
If you have any question, issue or suggestions, feel free to contact 
us with the emails listed in the paper.