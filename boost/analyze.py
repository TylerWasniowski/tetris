import numpy as np

print("Mean scores")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " + str(np.mean(a[0])))

print("Std deviation scores")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " + str(np.std(a[0])))

print("Mean number of moves")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " + str(np.mean(a[1])))

print("Std deviation number of moves")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " + str(np.std(a[1])))

print("Mean of single line clears")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " +
              str(np.mean(np.array(list(map(lambda x: list(x), a[2])))[:, 0])))

print("Mean of double line clears")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " +
              str(np.mean(np.array(list(map(lambda x: list(x), a[2])))[:, 1])))

print("Mean of triple line clears")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " +
              str(np.mean(np.array(list(map(lambda x: list(x), a[2])))[:, 2])))

print("Mean of quadruple line clears")
for n in range(2, 22):
    if not n == 11 and not n == 14:
        a = np.load("hmm_stats_r_1-" + str(n) + ".npy", allow_pickle=True)
        print(str(len(a[0])) + "               " + str(n) + "              " +
              str(np.mean(np.array(list(map(lambda x: list(x), a[2])))[:, 3])))
