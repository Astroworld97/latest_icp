from helpers import *
import ast
# # Open the file
# with open('exportDict.txt', 'r') as file:
#     # Initialize an empty dictionary
#     my_dict = {}
#     # Iterate over each line in the file
#     for line in file:
#         # Split the line into key and value using the colon separator
#         line = line.strip('\n')
#         key, value = line.split(':')
#         # Add the key value pair to the dictionary
#         my_dict[key] = value

d = txt_to_dict('exportDict.txt')

# for key in d:
#     print(key, d[key])

keys = list(d.keys())

plot(keys, d)
