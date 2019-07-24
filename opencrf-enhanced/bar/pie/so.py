import ctypes
import numpy as np

sb = ctypes.create_string_buffer
ll = ctypes.cdll.LoadLibrary
r = b"/home/jakob/experiment/OpenCRF/HardcodedPotentials/bar/rawdata/"
lib = ll("../../crflib.so")
print("we loaded the lib without crashing and dying, which is good")
with open("../rawdata/mapfile.txt","w") as mapname:
    print("../rawdata/maps/map0.txt",file=mapname)

lib.InitializeCRF.argtypes = [ctypes.c_char_p for i in range(5)]
lib.InitializeCRF.restype = ctypes.c_void_p
lib.train.argtypes = [ctypes.c_void_p]
lib.train.restype = None
lib.GetNNOutput.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.GetNNOutput.restype = None
lib.BackpropToNN.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.BackpropToNN.restype = None

model = lib.InitializeCRF(sb(b"0.001"), sb(r+b"fold-0-edges-ALT.txt"), sb(r+b"CRFmodel-0.txt"), sb(b"model-fold-0.txt"),sb(r+b"maps/map0.txt"))

for i in range(2):
    arr = np.random.random(7402*2)
    lib.GetNNOutput(model,ctypes.cast(arr.__array_interface__['data'][0],ctypes.POINTER(ctypes.c_double)), arr.shape[0])
    lib.train(model)
    arr2 = np.zeros(7402*2)
    print(arr2)
    print("retraining")
    lib.BackpropToNN(model,ctypes.cast(arr2.__array_interface__['data'][0],ctypes.POINTER(ctypes.c_double)),arr2.shape[0])
