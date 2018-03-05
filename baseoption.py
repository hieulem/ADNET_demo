import argparse
import os
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--root',type=str,default='/gpfs/projects/LynchGroup/')
        self.parser.add_argument('--train_fold',type=str,default='/gpfs/projects/LynchGroup/')
        self.parser.add_argument('--tif_fold',type=str,default='/gpfs/projects/LynchGroup/')
        self.parser.add_argument('--padding', type=int, default=1000)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
