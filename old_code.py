def construct_benchmark(n=32):
    """
    creates a Sequential Module that corresponds to the benchmark architecture
    conv3x3(n) -> conv3x3(n) -> max_pool -> conv3x3(1.5 n) -> conv3x3(1.5 n) -> max_pool -> conv3x3(2 n) ->  global_pool -> dense(10, nonlinearity = softmax)
    :param n: well, the n from the formula
    """

    model = nn.Sequential()
    model.add_module('conv_n_0', nn.Conv2d(1, n, kernel_size=(3, 3)))  # 26x26
    model.add_module('relu_n_0', nn.LeakyReLU(negative_slope=0.01))
    model.add_module('conv_n_1', nn.Conv2d(n, n, kernel_size=(3, 3)))  # 24x24
    model.add_module('relu_n_1', nn.LeakyReLU(negative_slope=0.01))
    #     model.add_module('bn1', nn.BatchNorm2d(n)) # 24x24
    model.add_module('pool_n', nn.MaxPool2d(kernel_size=(2, 2)))  # 12x12

    m = int(1.5 * n)

    model.add_module('conv_1_5n_0', nn.Conv2d(n, m, kernel_size=(3, 3)))  # 10x10
    model.add_module('relu_1_5n_0', nn.LeakyReLU(negative_slope=0.01))
    model.add_module('conv_1_5n_1', nn.Conv2d(m, m, kernel_size=(3, 3)))  # 8x8
    model.add_module('relu_1_5n_1', nn.LeakyReLU(negative_slope=0.01))
    #     model.add_module('bn2', nn.BatchNorm2d(m)) # 24x24
    model.add_module('pool_1_5n', nn.MaxPool2d(kernel_size=(2, 2)))  # 4x4

    model.add_module('conv_2n_0', nn.Conv2d(m, 2 * n, kernel_size=(3, 3)))  # 2x2
    model.add_module('relu_2n_0', nn.LeakyReLU(negative_slope=0.01))
    #     model.add_module('bn3', nn.BatchNorm2d(2*n)) # 24x24
    model.add_module('pool_2n', nn.MaxPool2d(kernel_size=(2, 2)))  # 1x1

    #     model.add_module('global_pool', GlobalMaxPool())
    #     model.add_module('dense0', nn.Linear(1, 10))

    model.add_module('global_pool', Flatten())
    model.add_module('dense0', nn.Linear(2 * n, 10))
    #     model.add_module('dense2_logits', nn.Linear(100, 10)) # logits for 10 classes
    model.apply(xavier_init)

    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)
    return model, opt

# ---------------------------------

model, opt = construct_benchmark(32)
cont = CudaTrainingContainer(model, opt, batch_size=512)
cont.compute_loss(X_train[:200], y_train[:200])
print(cont._batch_size)
cont.train(X_train, y_train, range(0,50), X_test, y_test)