cpu: tooooo slow
using cuda: ~0.89 s

constant memory和share memory都太小，无法把triangles或者vertices放进去

stream的可能会比较麻烦，还是先把包围盒做了

tris和vtxs不用每次collide()都拷贝到gpu一次,https://blog.csdn.net/venom_snake/article/details/120050822意思是不要在头文件里面include thrust: 提升不明显


把vertices的transform计算单独提出到preprocess_vtxs_kernel，减少重复计算（增加了显存消耗）: ~0.47 s


bounding sphere: ~0.13 s