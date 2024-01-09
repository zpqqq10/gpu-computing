success: 

1. cpu: ~122.22 s

2. using cuda: ~0.89 s

3. 把vertices的transform计算单独提出到preprocess_vtxs_kernel，减少重复计算（增加了显存消耗）: ~0.47 s

4. bounding sphere: ~0.13 s

5. pinned memory使用的空间有限，本项目需要大量gpu和cpu传输数据的情况比较少，因此提升较小: increase by ~0.001 s


fail:

1. constant memory和share memory都太小，无法把triangles或者vertices放进去

2. tris和vtxs不用每次collide()都拷贝到gpu一次,https://blog.csdn.net/venom_snake/article/details/120050822, 意思是不要在头文件里面include thrust: 提升不明显

3. stream估计意义不大，因为没有频繁的加载数据，体量比较大的数据只加载一次即可