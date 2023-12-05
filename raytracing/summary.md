1. 书本样例

发现使用书本给出的样例代码，难以产生区别，无论是否使用constant memory，时间都很接近。而且发现无论有没有使用constant memory，都有可能第一次运行时间很长，然后时间变低，如 200 个 spheres 时，第一次可能是 40-60ms，然后重复运行就是 2ms。因为是因为4090有做一些优化。

2. `cudaMemcpyToSymbol`

`cudaMemcpyToSymbol`的第一个变量好像一定要是直接声明的constant memory变量，中间要是经过函数传参好像会缺少某些属性导致报错，即使是声明参数时加上 `const` 或者`__constant__` 也不行

3. 使用constant memory和share memory

对于constant memory，尝试将小球数组存进constant memory，将小球运动的位移存到global memory。

对于share memory，在更新bitmap时将global memory中的小球位移数据存到share memory中。我认为这个任务不是很能体现share memory的优势，因为位移数组和小球数组是一对一的映射关系，而且更新bitmap时只需要读一次share memory。

以下表格为更新一次画面的用时，单位ms：

|             小球数量              | 1000 | 2000 |
| :-------------------------------: | :--: | :--: |
|          仅global memory          | 3.7  | 6.3  |
|         使用share memory          | 3.3  | 6.0  |
|        使用constant memory        | 2.7  | 4.5  |
| 使用share memory和constant memory | 2.7  | 3.7  |

4. block/thread

考虑warp size=32，在使用constant memory的条件下尝试了如下设置，单位ms：

| blockDim |      用时      |
| :------: | :------------: |
|  (1, 1)  |       72       |
| (16, 16) |      4.5       |
| (32, 32) |      4.2       |
| (64, 64) | 黑屏，原因未知 |

理论上，当blockDim为warp size的整数倍的时候性能会比较好，可以看到(32, 32)的用时比(16, 16)还短一点。