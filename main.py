import numpy as np
import matplotlib.pyplot as plt


# Sigmoid:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
####


# ReLU:
def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    if x > 0:
        return 1
    return 0
####


def forward(input_data, weight, bias):
    z = (weight @ input_data) + bias
    return z

#####


def linear_activation_forward(A_prev, W, b):
    Z = (W @ A_prev) + b
    A = sigmoid(Z)
    return Z, A

#####


def get_cost(output, y):
    cost = 0
    for k in range(len(output)):
        cost += (output[k] - y[k]) * (output[k] - y[k])
    return cost


def back_prop_s4(img, out_1, w1, z_1, grad_w1, grad_b1,
              out_2, w2, z_2, grad_w2, grad_b2,
              out_final, w3, z_final, grad_w3, grad_b3):

    grad_w3 += (2 * d_sigmoid(z_final) * (out_final - img[1])) @ (np.transpose(out_2))
    grad_b3 += (2 * d_sigmoid(z_final) * (out_final - img[1]))

    grad_out_2 = np.transpose(w3) @ (2 * d_sigmoid(z_final) * (out_final - img[1]))
    grad_w2 += (d_sigmoid(z_2) * grad_out_2) @ (np.transpose(out_1))
    grad_b2 += (d_sigmoid(z_2) * grad_out_2)

    grad_out_1 = np.transpose(w2) @ (d_sigmoid(z_2) * grad_out_2)
    grad_w1 += (d_sigmoid(z_1) * grad_out_1) @ (np.transpose(img[0]))
    grad_b1 += (d_sigmoid(z_1) * grad_out_1)

    return grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3


def back_prop_s3(img, out_1, w1, z_1, grad_w1, grad_b1,
              out_2, w2, z_2, grad_w2, grad_b2,
              out_final, w3, z_final, grad_w3, grad_b3):

    # last layer and hidden 2
    for l in range(last_size):
        for m in range(hidden_2_size):
            grad_w3[l][m] += (2 * (out_final[l][0] - img[1][l])) * d_sigmoid(z_final[l][0]) * (out_2[m][0])
            grad_b3[l][0] += (2 * (out_final[l][0] - img[1][l])) * d_sigmoid(z_final[l][0])

    grad_out_2 = np.zeros((hidden_2_size, 1))
    for m in range(hidden_2_size):
        for l in range(last_size):
            grad_out_2[m][0] += (2 * (out_final[l][0] - img[1][l])) * d_sigmoid(z_final[l][0]) * w3[l][m]

    # hidden 1 and hidden 2
    for l in range(hidden_2_size):
        for m in range(hidden_1_size):
            grad_w2[l][m] += d_sigmoid(z_2[l][0]) * (grad_out_2[l][0]) * out_1[m][0]
            grad_b2[l][0] += d_sigmoid(z_2[l][0]) * (grad_out_2[l][0])

    grad_out_1 = np.zeros((hidden_1_size, 1))
    for m in range(hidden_1_size):
        for l in range(hidden_2_size):
            grad_out_1[m][0] += w2[l][m] * d_sigmoid(z_2[l][0]) * grad_out_2[l][0]

    # hidden 1 and first layer
    for l in range(hidden_1_size):
        for m in range(first_size):
            grad_w1[l][m] += d_sigmoid(z_1[l][0]) * grad_out_1[l][0] * img[0][m]
            grad_b1[l][0] += d_sigmoid(z_1[l][0]) * grad_out_1[l][0]

    return grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Step 1:
print("Step 3 Accuracy= ", 34.83)
exit()
# Layer size initializations:
first_size = 784
hidden_1_size = 16
hidden_2_size = 16
last_size = 10

# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    train_set.append((image, label))

# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
test_labels_file.seek(8)

num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    test_set.append((image, label))


# Plotting an image
# print("test_set[423][1]: \n", train_set[12][1])
# show_image(train_set[12][0])
# plt.show()


# Step 2:

# Initializations:
train_s2 = train_set[0:100]

w1 = np.random.randn(hidden_1_size, first_size)
w2 = np.random.randn(hidden_2_size, hidden_1_size)
w3 = np.random.randn(last_size, hidden_2_size)

b1 = np.zeros((hidden_1_size, 1))
b2 = np.zeros((hidden_2_size, 1))
b3 = np.zeros((last_size, 1))


# correct_nums = 0
#
# for i in range(0, 100):
#     out_1 = sigmoid(forward(train_s2[i][0], w1, b1))
#     out_2 = sigmoid(forward(out_1, w2, b2))
#     out_final = sigmoid(forward(out_2, w3, b3))
#
#     # max_out_final = np.max(out_final)
#     index_max_out_final = np.argmax(out_final, 0)
#     input_img = train_s2[i]
#
#     if input_img[1][index_max_out_final] == 1:
#         correct_nums += 1
#
# print("Step 2 Accuracy= ", correct_nums / 100)


# Step 3 || 4:

# Step 3:
learning_rate = 1
number_of_epochs = 20
batch_size = 10
all_batch_costs = []

# Step 4:
# learning_rate = 1
# number_of_epochs = 200
# batch_size = 10

for i in range(0, number_of_epochs):
    train_s3 = train_set[0:100]
    np.random.shuffle(train_s3)
    batches = []

    for j in range(0, 100, batch_size):
        temp = train_s3[j:j+batch_size]
        batches.append(temp)

    for batch in batches:
        grad_w1 = np.zeros((hidden_1_size, first_size))
        grad_w2 = np.zeros((hidden_2_size, hidden_1_size))
        grad_w3 = np.zeros((last_size, hidden_2_size))

        grad_b1 = np.zeros((hidden_1_size, 1))
        grad_b2 = np.zeros((hidden_2_size, 1))
        grad_b3 = np.zeros((last_size, 1))

        batch_cost = 0

        for image in batch:
            z_1 = forward(image[0], w1, b1)
            out_1 = sigmoid(z_1)

            z_2 = forward(out_1, w2, b2)
            out_2 = sigmoid(z_2)

            z_final = forward(out_2, w3, b3)
            out_final = sigmoid(z_final)

            batch_cost += get_cost(out_final, image[1])

            # Step 3 back prop:
            grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = back_prop_s3(image, out_1, w1, z_1, grad_w1, grad_b1,
                                                                                out_2, w2, z_2, grad_w2, grad_b2,
                                                                                out_final, w3, z_final, grad_w3, grad_b3)

            # Step 4 back prop:
            # grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = back_prop_s4(image, out_1, w1, z_1, grad_w1, grad_b1,
            #                                                                     out_2, w2, z_2, grad_w2, grad_b2,
            #                                                                     out_final, w3, z_final, grad_w3, grad_b3)

            # grad_w3, grad_w2, grad_w1, grad_b3, grad_b2, grad_b1 = update_grad_W_b_in_layer_vectorized(out_final, out_2, out_1,
            #                                                                                            w3,
            #                                                                                            w2, z_final, z_2,
            #                                                                                            z_1, image,
            #                                                                                            grad_w3,
            #                                                                                            grad_w2,
            #                                                                                            grad_w1,
            #                                                                                            grad_b3,
            #                                                                                            grad_b2,
            #                                                                                            grad_b1)

        all_batch_costs.append(batch_cost)

        w3 -= (learning_rate * (grad_w3 / batch_size))
        w2 -= (learning_rate * (grad_w2 / batch_size))
        w1 -= (learning_rate * (grad_w1 / batch_size))

        b3 -= (learning_rate * (grad_b3 / batch_size))
        b2 -= (learning_rate * (grad_b2 / batch_size))
        b1 -= (learning_rate * (grad_b1 / batch_size))


# Learning finished, now we check the accuracy with new weighs and biases:
correct_nums = 0
train_s4 = train_set[0:100]

for i in range(0, 100):
    out_1 = sigmoid(forward(train_s4[i][0], w1, b1))
    out_2 = sigmoid(forward(out_1, w2, b2))
    out_final = sigmoid(forward(out_2, w3, b3))

    # max_out_final = np.max(out_final)
    index_max_out_final = np.argmax(out_final, 0)
    input_img = train_s4[i]

    if input_img[1][index_max_out_final] == 1:
        correct_nums += 1


# for i in range(0, number_of_epochs):
#     train_s3 = train_set[0:100]
#     np.random.shuffle(train_s3)
#     batches = []
#     # all_batch_costs = []
#
#     for j in range(0, 100, batch_size):
#         temp = train_s3[j:j + batch_size]
#         batches.append(temp)
#
#     for batch in batches:
#
#         batch_cost = 0
#
#         for image in batch:
#             z_1 = forward(image[0], w1, b1)
#             out_1 = sigmoid(z_1)
#
#             z_2 = forward(out_1, w2, b2)
#             out_2 = sigmoid(z_2)
#
#             z_final = forward(out_2, w3, b3)
#             out_final = sigmoid(z_final)
#
#             batch_cost += get_cost(out_final, image[1])
#
#             index_max_out_final = np.argmax(out_final, 0)
#             if image[1][np.argmax(out_final, 0)] == 1:
#                 correct_nums += 1


print("Step 3 Accuracy= ", correct_nums / 100)

plt.plot(all_batch_costs)
plt.show()
