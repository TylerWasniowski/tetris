import numpy as np

print("Mean scores")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(n) + "                            " + str(np.mean(a[0])))

print("Std deviation scores")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(n) + "                            " + str(np.std(a[0])))

print("Mean number of moves")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(n) + "                            " + str(np.mean(a[1])))

print("Std deviation number of moves")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(n) + "                            " + str(np.std(a[1])))
