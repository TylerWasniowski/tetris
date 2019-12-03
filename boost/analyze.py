import numpy as np

for n in range(2, 21):
    if not n == 11:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.mean(a[0])))
        finally:
            print()

for n in range(2, 21):
    if not n == 11:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.max(a[0])))
        finally:
            print()

for n in range(2, 21):
    if not n == 11:
        try:
            a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
            print(str(n) + "                            " + str(np.std(a[0])))
        finally:
            print()
