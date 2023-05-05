import copy

original_dict = {'a': 1, 'b': {'c': 2, 'd': 3}}

# Make a deep copy of the original dictionary
copied_dict = copy.deepcopy(original_dict)

# Modify the nested dictionary in the copied dictionary
copied_dict['b']['c'] = 4

# Check that the original dictionary is unchanged
print(original_dict)  # {'a': 1, 'b': {'c': 2, 'd': 3}}

# Check that the copied dictionary has been modified
print(copied_dict)  # {'a': 1, 'b': {'c': 4, 'd': 3}}
