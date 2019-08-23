# coding: utf-8
import pytest
import spacy
import pytest

from num_fh import NFH


@pytest.fixture(scope='function')
def nlp():
    return spacy.load('en_core_web_sm')


def test_integration(nlp):
    emoji = NFH(nlp)
    nlp.add_pipe(emoji, last=True)
    assert nlp.pipe_names[-1] == 'num_fh'


def test_simple(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("I bought 5 apples but got only 4")
    assert not doc[2]._.is_nfh
    assert doc[-1]._.is_nfh

    assert not doc[-1]._.is_deter_nfh
    assert len(doc._.nfh) == 1


def test_multiple_fh(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("I told you two, that only one of them is the one who will get 2 or 3 icecreams")
    assert doc[16]._.is_nfh
    assert not doc[18]._.is_nfh
    assert doc[3]._.is_deter_nfh
    assert not doc[16]._.is_deter_nfh
    assert len(doc._.nfh) == 4


def test_span(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("How much was it? Two hundred, but I'll tell him its fifty.")
    assert doc[5:7]._.is_nfh
    assert doc[5]._.is_nfh
    assert doc[6]._.is_nfh
    print(doc[5:6]._.is_nfh)
    print(doc[6:7]._.is_nfh)
    assert not doc[5:6]._.is_nfh
    assert not doc[6:7]._.is_nfh

    assert doc[5]._.nfh_head == 'CURRENCY'
    assert doc[6]._.nfh_head == 'CURRENCY'

    assert doc[5]._.is_implicit
    assert doc[6]._.is_implicit

    assert not doc[5]._.is_deter_nfh
    assert not doc[6]._.is_deter_nfh


def test_year(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("Do you know I haven't been to the theatre since, eh... '96.")
    assert doc[-2]._.is_nfh
    assert doc[-2]._.nfh_head == 'YEAR'


def test_age(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("I probably feel worse than Demi Moore did when she turned 50.")
    assert doc[-2]._.is_nfh
    assert doc[-2]._.nfh_head == 'AGE'


def test_time(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("Let's meet at 5 in the morning?")
    assert doc[4]._.is_nfh
    assert doc[4]._.nfh_head == 'TIME'


def test_people(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("Are you two done with that helium?")
    assert doc[2]._.is_nfh
    assert doc[2]._.nfh_head == 'PEOPLE'


def test_other(nlp):
    nfh = NFH(nlp)
    nlp.add_pipe(nfh, last=True)

    doc = nlp("My manager says I'm a perfect 10!")
    assert doc[-2]._.is_nfh
    assert doc[-2]._.nfh_head == 'OTHER'
