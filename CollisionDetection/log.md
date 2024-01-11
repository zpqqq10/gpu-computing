success: 

1. cpu: ~122.22 s

2. using cuda: ~0.89 s

3. 把vertices的transform计算单独提出到preprocess_vtxs_kernel，减少重复计算（增加了显存消耗）: ~0.47 s

4. bounding sphere: ~0.134 s

5. pinned memory使用的空间有限，本项目需要大量gpu和cpu传输数据的情况比较少，因此提升较小: ~0.133 s

6. 把spheres的transform提取到了preprocess_spheres_kernel，减少重复计算（增加了显存消耗）: ~0.129 s

7. sphere相交计算中把速度较慢的sqrt等效替换掉了，加速: ~0.075 s

8. 用aabb替代bounding sphere，估计是距离的计算比较耗时而且球形不如立方体贴合三角形面片: ~0.034 s
所有的源文件都要以cu结尾，不然nvcc可能会以gcc编译头文件，导致头文件的对象不能有device_vector成员

9. 加了bvh的排序部分后，速度又快了一点: ~0.030 s

10. using bvh: ~0.003 s


fail:

1. constant memory和share memory都太小，无法把triangles或者vertices放进去

2. tris和vtxs不用每次collide()都拷贝到gpu一次,https://blog.csdn.net/venom_snake/article/details/120050822, 意思是不要在头文件里面include thrust: 提升不明显

3. stream估计意义不大，因为没有频繁的加载数据，体量比较大的数据只加载一次即可