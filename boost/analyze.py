import numpy as np

print("Mean scores")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.mean(a[0])))
        finally:
            print()

print("Std deviation scores")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.std(a[0])))
        finally:
            print()

print("Mean number of moves")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.mean(a[1])))
        finally:
            print()

print("Std deviation number of moves")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.std(a[1])))
        finally:
            print()


