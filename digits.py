from neural import *
from PIL import Image

# Load neural network
network = Network.LoadNetwork('digits.nn')

# Loop until the user decides to exit
answer = input('Enter the name of a 28x28 image file to determine which digit is drawn, or type \'exit\' to exit.\n')
while answer != 'exit':
    try:
        if answer == '':
            raise FileNotFoundError
        # Open the image
        with Image.open(answer) as image:
            # Check that the image has correct size
            if image.size == (28, 28):
                # Get brightness data from image
                image = image.convert(mode='L')
                data = [(255 - i)/255 for i in image.getdata()]
                # Feed data into neural network and print the result
                results = [round(i) for i in network.FeedForward(data)]
                if results.count(1) == 1:
                    print('The image contains the digit {0}.'.format(results.index(1)))
                else:
                    print('The image contains an unknown digit.')
            else:
                print('The image is of incorrect size.')
    except:
        print('Invalid file, please try again.')
    answer = input()
print()