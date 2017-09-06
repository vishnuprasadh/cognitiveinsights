import itertools
from itertools import groupby
from operator import itemgetter
from collections import OrderedDict
dict1 = dict()
dict1['A']=1
dict1['B']=2
dict1['C']=1
dict1['D']=3
dict1['E']=1
dict1['F']=4

print(dict1.items (sorted(dict1.values())))

for k,v in groupby(dict1.get(sorted(dict1.values())), key=lambda x : x ):
    print(k,list(v))
