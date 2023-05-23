import pickle

# some_list = [float(i) for i in range(10000000)]

# metadata_file = "test.pickle"
# with open(metadata_file, 'wb') as handle:
#     pickle.dump(some_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

a = pickle.load(open("test.pickle", "rb"))

print("hello")
b = [a[i]*66 for i in range(len(a))]
print("the end")