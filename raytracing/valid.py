threadIdx = {
    'x': 0,
    'y': 0,
}

blockIdx = {
    'x': 0,
    'y': 0,
}

blockDim = {
    'x': 2,
    'y': 3,
}

gridDim = {
    'x': 2,
    'y': 2,
}

for i in range(gridDim['x']): 
    for j in range(gridDim['y']): 
        for k in range(blockDim['x']):
            for t in range(blockDim['y']):
                x = k + i * blockDim['x']
                y = t + j * blockDim['y']
                offset = x + y * gridDim['x'] * blockDim['x']
                
                blockId = i + j * gridDim['x']
                threadId = blockId * (blockDim['x'] * blockDim['y']) + (t * blockDim['x']) + k
                
                print(offset, '|', threadId)


# x = threadIdx['x'] + blockIdx['x'] * blockDim['x']
# y = threadIdx['y'] + blockIdx['y'] * blockDim['y']
# offset = x + y * gridDim['x'] * blockDim['x']

# blockId = blockIdx['x'] + blockIdx['y'] * gridDim['x']
# threadId = blockId * (blockDim['x'] * blockDim['y']) + (threadIdx['y'] * blockDim['x']) + threadIdx['x']

# print(offset)
# print(threadId)