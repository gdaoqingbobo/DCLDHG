


input_file_path = '../Data/Micro/alldata.txt'  #
train_file_path = '../Data/Micro/train.csv'    #
test_file_path = '../Data/Micro/test.csv'      #


with open(input_file_path, 'r') as file:
    lines = file.readlines()

#
train_lines = lines[:2210]  #
test_lines = lines[2210:]   #

#
train_data = [",".join(line.split()) for line in train_lines]
test_data = [",".join(line.split()) for line in test_lines]


with open(train_file_path, 'w') as train_file:
    train_file.write("\n".join(train_data))


with open(test_file_path, 'w') as test_file:
    test_file.write("\n".join(test_data))

print("Train and test files generated successfully.")
