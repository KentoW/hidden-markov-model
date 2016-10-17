# -*- coding: utf-8 -*-
# ベイジアン隠れマルコフモデル(hidden markov model)
import sys
import math
import random
import argparse
import scipy.special
from collections import defaultdict

class HMM:
    def __init__(self, data):
        self.corpus_file = data
        self.target_word = defaultdict(int)
        self.corpus = []
        comment = ""
        for strm in open(data, "r"):




def main(args):
    hmm = HMM(args.data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01, type=float, help="hyper parameter alpha")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01, type=float, help="hyper parameter beta")
    parser.add_argument("-k", "--K", dest="K", default=10, type=int, help="number of hidden variable")
    parser.add_argument("-n", "--N", dest="N", default=1000, type=int, help="max iteration")
    parser.add_argument("-c", "--converge", dest="converge", default=0.01, type=str, help="converge")
    parser.add_argument("-d", "--data", dest="data", default="data.txt", type=str, help="training data")
    args = parser.parse_args()
    main(args)
