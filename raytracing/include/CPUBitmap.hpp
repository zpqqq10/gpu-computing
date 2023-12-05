#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include "gl_helper.h"

struct CPUBitmap {
  unsigned char *pixels; /*像素点的总个数*/
  int x, y;              /*图像的长宽*/
  void *dataBlock;       /*  */
  void (*bitmapExit)(void *);

  CPUBitmap(int width, int height, void *d = NULL) {
    // 4: RGBA
    pixels = new unsigned char[width * height * 4];
    x = width;     /*图像的宽*/
    y = height;    /*图像的高*/
    dataBlock = d; /* */
  }

  ~CPUBitmap() {
    /*删除像素点*/
    delete[] pixels;
  }

  /*取得所有像素点*/
  unsigned char *get_ptr(void) const { return pixels; }

  /*取得图片总大小*/
  long image_size(void) const { return x * y * 4; }

  void display_and_exit(void (*e)(void *) = NULL) {
    CPUBitmap **bitmap = get_bitmap_ptr();
    *bitmap = this;
    bitmapExit = e;

    // a bug in the Windows GLUT implementation prevents us from
    // passing zero arguments to glutInit()
    int c = 1;
    char *dummy = "";

    /*glutInit,对 GLUT (OpenGl
     * 里面的一个工具包，包含很多函数)进行初始化,这个函数必须在其它的
     * GLUT使用之前调用一次。其格式比较死板,一般照抄这句glutInit(&argc,
     * argv)就可以了*/
    glutInit(&c, &dummy);
    /*设置显示方式,其中 GLUT_RGBA 表示使用 RGBA
     * 颜色,与之对应的还有GLUT_INDEX(表示使用索引颜色) ；GLUT_SINGLE
     * 表示使用单缓冲,。与之对应的还有 GLUT_DOUBLE(使用双缓冲)。*/
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize(x, y);
    glutCreateWindow("bitmap");
    glutKeyboardFunc(Key);
    glutDisplayFunc(Draw);
    glutMainLoop();
  }

  // static method used for glut callbacks
  static CPUBitmap **get_bitmap_ptr(void) {
    static CPUBitmap *gBitmap;
    return &gBitmap;
  }

  // static method used for glut callbacks
  static void Key(unsigned char key, int x, int y) {

    switch (key) {
    case 27:
      // exit if esc is pressed
      CPUBitmap *bitmap = *(get_bitmap_ptr());
      if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL)
        bitmap->bitmapExit(bitmap->dataBlock);
      exit(0);
    }
  }

  // static method used for glut callbacks

  /* 画图 */
  static void Draw(void) {
    CPUBitmap *bitmap = *(get_bitmap_ptr());

    /*设置背景颜色*/
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE,
                 bitmap->pixels);
    glFlush();
  }
};

#endif // __CPU_BITMAP_H__