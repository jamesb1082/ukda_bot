def test_imports(): 
    try: 
        import numpy 
        import scipy
        import pandas 
        import sklearn
        import chatterbot
        import matplotlib
        import seaborn 
        import tabulate
        import gensim
        import tensorflow
        import keras
        import h5py
        import pickle
        import progressbar
        import flask
    except ImportError: 
        return False
    return True

assert test_imports() == True
print("Test passed") 
