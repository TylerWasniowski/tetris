from hmmlearn import hmm
import math
import os.path
import pickle


HMM_FILE_NAME_BASE = "hmm_"
HMM_FILE_NAME_EXTENSION = "_2c.hmm"


def load_hmm(n=8, compressed=True):
    if compressed:
        extension = "_2c.hmm"
    else:
        extension = "_2r.hmm"

    file_name = __get_file_name(n, extension=extension)
    if not os.path.exists(file_name):
        return None

    with open(file_name, "rb") as file:
        return pickle.load(file)


# Saves hmm if it scores better on observations than already saved hmm
def maybe_save_hmm(model, observations):
    n = model.n_components
    saved_model = load_hmm(n)
    if saved_model:
        saved_model_score = saved_model.score(observations)
        model_score = model.score(observations)

        if saved_model_score >= model_score:
            return

    print("Saving hmm")
    file_name = __get_file_name(n)
    with open(file_name, "wb") as file:
        pickle.dump(model, file)


def train_hmm(observations, n=8, n_iter=350, restarts=1, verbose=False):
    best_model = None
    best_score = -math.inf
    for i in range(1, restarts + 1):
        model = hmm.MultinomialHMM(n_components=n, n_iter=n_iter, tol=0.1, verbose=verbose)
        model.fit(observations)
        score = model.score(observations)
        if score > best_score:
            print("Found model with better score:", score, "(+", str(score - best_score) + ")")
            best_model = model
            best_score = score
            maybe_save_hmm(best_model, observations)
        print("Finished restart:", i)
        print("Score this restart:", score)
        print("Best score so far:", best_score)
        print()

    return best_model


def __get_file_name(n, extension=HMM_FILE_NAME_EXTENSION):
    return HMM_FILE_NAME_BASE + str(n) + extension
