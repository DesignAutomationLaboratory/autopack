import pickle


def save_dataset(ds, path):
    with open(path, "wb") as file:
        pickle.dump(ds, file)
    return path


def load_dataset(path):
    with open(path, "rb") as file:
        ds = pickle.load(file)
    return ds
