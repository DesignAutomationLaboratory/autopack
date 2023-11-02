import facit


def save_dataset(ds, path):
    return facit.dump_zarr(ds, path)


def load_dataset(path):
    return facit.load_zarr(path)
