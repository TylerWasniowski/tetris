import numpy as np

for n in range(2, 22):
    print("Mean scores")
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.mean(a[0])))
        finally:
            print()

for n in range(2, 22):
    print("Std deviation scores")
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.std(a[0])))
        finally:
            print()

for n in range(2, 22):
    print("Mean number of moves")
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.mean(a[1])))
        finally:
            print()

for n in range(2, 22):
    print("Std deviation number of moves")
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.std(a[1])))
        finally:
            print()


