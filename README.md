## 1 Julia Set

要求基本就是能跑就行

```bash
cd julia
mkdir build && cd build
cmake ..
make
./gmain # or ./cmain
```

## 2 raytracing

constant memory和share memory

```bash
cd raytracing
mkdir build && cd build
cmake .. && make
./main
```

## 3 stream

学会使用stream，用nsys分析

```bash
cd stream
mkdir build && cd build
cmake .. && make
./main_stream
```

## 4 Collision Detection

想办法加速就完了，最后效果最明显的是bvh

最关键的是学会了用thrust，感觉让cuda编程从c原生时代进入了cpp时代

```bash
cd CollisionDetection
mkdir build && cd build
cmake .. && make
./main ../data/my-bunny.obj ../data/my-bunny.obj
```
