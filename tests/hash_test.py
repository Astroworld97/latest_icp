import hashlib

my_string = "My String!"

my_string_bits  = my_string.encode('utf-8')

secret_thing = hashlib.sha256(my_string_bits)

hash = secret_thing.hexdigest()

print(hash)