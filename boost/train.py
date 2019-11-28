import argparse

import ai
import observations


parser = argparse.ArgumentParser()
parser.add_argument("--n", "-n", help="Number of hidden states", default=8, type=int)
parser.add_argument("--iters", "-i", help="Iterations to do with each set of initial conditions", default=350, type=int)
parser.add_argument("--restarts", "-r", help="Times to restart with new initial conditions", default=1, type=int)
parser.add_argument("--verbose", "-v", help="Print score each iter", default=False, type=bool)
args = parser.parse_args()

ai.train_hmm(observations.compressed(), n=args.n, n_iter=args.iters, restarts=args.restarts, verbose=args.verbose)
