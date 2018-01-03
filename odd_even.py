from neural import *

# Converts a number to a 16-bit binary representation
def NumToBin(x):
    res = list(map(int, list(bin(x)[2:])))
    res = [0]*(16-len(res)) + res
    return res

# Create training data set
inputs_set = []
expected_set = []
even = True
for i in range(1000):
    inputs_set.append(NumToBin(i))
    expected_set.append([1] if even else [0])
    even = not even

# Create and train neural network
network = Network(3, [16, 20, 1])
network.Train(inputs_set, expected_set, 3)

# Loop until the user wants to exit
answer = input('Enter a number between 0 and 65535 to test whether it is even or odd, or type anything else to exit.\n')
while (answer.isdigit()) and (0 <= int(answer) < 65536):
    result = network.FeedForward(NumToBin(int(answer)))[0]
    print('Even' if result > 0.5 else 'Odd')
    answer = input()
print()
