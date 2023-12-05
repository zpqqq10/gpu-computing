1. constant memory

在书本给出的 sample 中，难以产生区别，而且发现无论有没有加 constant，都有可能第一次运行时时间很长，然后时间就很低，如 200 个 spheres 时，第一次可能是 40-60ms，然后重复运行就是 2ms

在代码中，如果仅仅使用 costant memory，可以将每一帧的时间从大约 2.7-2.8ms 降到 2.3-2.4ms

cudaMemcpyToSymbol 的第一个变量好像一定要是直接声明的 costant 变量，中间要是经过传参好像会缺少某些属性导致报错，即使是声明参数时加上 `const` 或者`__constant__` 也不行

2. comparison
1000
shared: 3.3
constant: 2.7
nothing: 3.7
both: 2.7

2000
both: 3.7
shared: 6.0
constant: 4.5
nothing: 6.3

32x32 with constant: 4.2

1x1 with constant: 72