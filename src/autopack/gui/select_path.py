import pickle
import tkinter as tk
from tkinter import filedialog


def select_file_path():
    window = tk.Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()  # this supress the tk window
    filename = filedialog.askopenfilename(
        parent=window,
        initialdir="",
        title="Select A File",
        filetypes=(("Text files", "*.txt"), ("All files", "*")),
    )
    return filename


def select_save_file_path():
    window = tk.Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()  # this supress the tk window
    filename = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("json files", "*.json")],
        title="Choose filename",
    )
    return filename


def load_json(path):
    with open(path, "rb") as file:
        s1_new = pickle.load(file)
    return s1_new


def dump_json(to_save, path):
    with open(path, "wb") as file:
        pickle.dump(to_save, file)
