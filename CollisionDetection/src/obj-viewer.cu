//**************************************************************************************
//  Copyright (C) 2022 - 2024, Min Tang (tang_m@zju.edu.cn)
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER
//  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <GL/glh_glut.h>
#include <omp.h>
#include <stdio.h>
#include "forceline.h"

//****************************** add for linux ******************************//
#include <time.h>
// 返回自系统开机以来的毫秒数（tick）
unsigned long GetTickCount()
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
//****************************** add for linux ******************************//


bool b[256];
int win_w = 512, win_h = 512;

using namespace glh;
glut_simple_mouse_interactor object;
void CaptureScreen(int, int);

char *dataPath;
int stFrame = 0;
int pFrame = stFrame;

// for sprintf
#pragma warning(disable : 4996)

extern void initModel(const char *, const char *);
extern void quitModel();
extern void drawModel(bool, bool, bool, bool, int);
extern bool dynamicModel(char *, bool, bool);
extern bool exportModel(const char *);
extern bool importModel(const char *);
extern void checkCollision();

static int level = -1;

float lightpos1[4] = {13, 10.2, 3.2, 0};
float lightpos[4] = {0, 0, 0, 0};

// check for OpenGL errors
void checkGLError() {
  GLenum error;
  while ((error = glGetError()) != GL_NO_ERROR) {
    char msg[512];
    sprintf(msg, "error - %s\n", (char *)gluErrorString(error));
    printf(msg);
  }
}

void initSetting() { b['9'] = false; }

void initOpengl() {
  glClearColor(1.0, 1.0, 1.0, 1.0);

  // initialize OpenGL lighting
  GLfloat lightPos[] = {10.0, 10.0, 10.0, 0.0};
  GLfloat lightAmb[4] = {0.0, 0.0, 0.0, 1.0};
  GLfloat lightDiff[4] = {1.0, 1.0, 1.0, 1.0};
  GLfloat lightSpec[4] = {1.0, 1.0, 1.0, 1.0};

  glLightfv(GL_LIGHT0, GL_POSITION, &lightpos1[0]);
  glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
  glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpec);

  // glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL_EXT,
  // GL_SEPARATE_SPECULAR_COLOR_EXT);
  GLfloat black[] = {0.0, 0.0, 0.0, 1.0};
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
}

void drawText(const char *text, int length, int x, int y) {
  glDisable(GL_LIGHTING);
  glColor3f(1, 0, 0);

  glMatrixMode(GL_PROJECTION);
  double *matrix = new double[16];
  glGetDoublev(GL_PROJECTION_MATRIX, matrix);
  glLoadIdentity();
  glOrtho(0, 800, 0, 600, -5, 5);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glPushMatrix();
  glLoadIdentity();
  glRasterPos2i(x, y);
  for (int i = 0; i < length; i++) {
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, (int)text[i]);
    // printf("Hello world!\n");
  }
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixd(matrix);
  glMatrixMode(GL_MODELVIEW);

  glEnable(GL_LIGHTING);
}

static char fpsBuffer[512];

void CalculateFrameRate() {
  static float framesPerSecond = 0.0f;
  static float lastTime = 0.0f;
  static bool first = true;
  if (first) {
    lastTime = GetTickCount() * 0.001f;
    first = false;
  }
  float currentTime = GetTickCount() * 0.001f;

  ++framesPerSecond;
  float delta = currentTime - lastTime;
  if (delta > 1.0f) {
    lastTime = currentTime;
    sprintf(fpsBuffer, "FPS: %d", int(ceil(framesPerSecond)));
    framesPerSecond = 0;
  }
}

const char *usageLn1 = "d - toggle animation";
const char *usageLn2 = "t - show/hide collision";
const char *usageLn3 = "p - show/hide bodies";
const char *usageLn4 = "-/= - show BVH nodes";
const char *usageLn5 = "s - show modelA only";

void updateFPS() {
  CalculateFrameRate();
  // glutSetWindowTitle(fpsBuffer);
  drawText(fpsBuffer, strlen(fpsBuffer), 5, win_h + 20);
  drawText(usageLn1, strlen(usageLn1), 5, win_h + 5);
  drawText(usageLn2, strlen(usageLn2), 5, win_h - 10);
  drawText(usageLn3, strlen(usageLn3), 5, win_h - 25);
  drawText(usageLn4, strlen(usageLn4), 5, win_h - 40);
  drawText(usageLn5, strlen(usageLn5), 5, win_h - 55);
}

void begin_window_coords() {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, win_w, 0.0, win_h, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void end_window_coords() {
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void drawGround() {
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

  glBegin(GL_QUADS);
  glColor3f(1.f, 0.f, 0.f);
  glVertex3f(20, 0, 20);
  glVertex3f(-20, 0, 20);
  glVertex3f(-20, 0, -20);
  glVertex3f(20, 0, -20);
  glEnd();

  glDisable(GL_COLOR_MATERIAL);
}

void draw() {
  glPushMatrix();
  // glRotatef(-90, 1, 0, 0);

  drawModel(!b['t'], !b['p'], b['s'], b['e'], level);

  glPopMatrix();
}

static bool ret = false;

void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glShadeModel(GL_SMOOTH);

  if (!b['b']) {
    // gradient background
    begin_window_coords();
    glBegin(GL_QUADS);
    glColor3f(0.2, 0.4, 0.8);
    glVertex2f(0.0, 0.0);
    glVertex2f(win_w, 0.0);
    glColor3f(0.05, 0.1, 0.2);
    glVertex2f(win_w, win_h);
    glVertex2f(0, win_h);
    glEnd();
    updateFPS();
    end_window_coords();
  }

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  object.apply_transform();

  // draw scene
  if (b['w'])
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  // draw scene
  /*	if (b['l'])
                  glDisable(GL_LIGHTING);
          else
  */
  glEnable(GL_LIGHTING);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);

  draw();

  glutSwapBuffers();

  if (b['x'] && ret) {
    CaptureScreen(512, 512);
  }
}

void key5();
void key4();
void key6();
void key1();
void key2();

void idle() {
  if (b[' '])
    object.trackball.increment_rotation();

  if (b['d']) {
    key1();
  }

  glutPostRedisplay();
}

void key2() {
  static int idx = 0;
  char buffer[512];
  sprintf(buffer, "c:\\temp\\min-%05d.scn", idx++);
  exportModel(buffer);
}

void key3() {
  static int idx = 0;
  char buffer[512];
  sprintf(buffer, "c:\\temp\\min-%05d.scn", idx++);
  importModel(buffer);
}

void key4() { checkCollision(); }

void key1() {
  dynamicModel(dataPath, b['o'], false);
  // key4();
}

void quit() {
  quitModel();
  exit(0);
}

void printLight() {
  printf("Light: %f, %f, %f, %f\n", lightpos[0], lightpos[1], lightpos[2],
         lightpos[3]);
}

void updateLight() { glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]); }

void endCapture() {}

void key(unsigned char k, int x, int y) {
  b[k] = !b[k];

  switch (k) {
  case 27:
  case 'q':
    quit();
    break;

  case 'x': {
    if (b['x'])
      printf("Starting screen capturing.\n");
    else
      printf("Ending screen capturing.\n");

    break;
  }

  case '1':
    key1();
    break;

  case '2':
    key2();
    break;

  case '3':
    key3();
    break;

  case '4':
    key4();
    break;

  case '=':
    level++;
    break;

  case '-':
    level--;
    break;
  }

  object.keyboard(k, x, y);
  glutPostRedisplay();
}

void resize(int w, int h) {
  if (h == 0)
    h = 1;

  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.1, 500.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  object.reshape(w, h);

  win_w = w;
  win_h = h;
}

void mouse(int button, int state, int x, int y) {
  object.mouse(button, state, x, y);
}

void motion(int x, int y) { object.motion(x, y); }

void main_menu(int i) { key((unsigned char)i, 0, 0); }

void initMenu() {
  glutCreateMenu(main_menu);
  glutAddMenuEntry("Toggle animation [d]", 'd');
  glutAddMenuEntry("Toggle obb/aabb [o]", 'o');
  glutAddMenuEntry("========================", '=');
  glutAddMenuEntry("Toggle rebuild/refit  (aabb) [r]", 'r');
  glutAddMenuEntry("Increasing boxes level(aabb) [=]", '=');
  glutAddMenuEntry("Decreasing boxes level(aabb) [-]", '-');
  glutAddMenuEntry("========================", '=');
  glutAddMenuEntry("Toggle wireframe [w]", 'w');
  glutAddMenuEntry("Toggle lighting [l]", 'l');
  glutAddMenuEntry("Toggle avi recording [x]", 'x');
  glutAddMenuEntry("Save camera[s]", 's');
  glutAddMenuEntry("Reset camera[t]", 't');
  glutAddMenuEntry("========================", '=');
  glutAddMenuEntry("Quit/q [esc]", '\033');
  glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void usage_self_cd() {
  printf("Keys:\n");
  printf("4 - Check collisions.\n");
  printf("t - Toggle display colliding.\n");
  printf("d - Toggle animation.\n");
  printf("q/ESC - Quit.\n\n");
  printf("Mouse:\n");
  printf("Left Btn - Obit.\n");
  printf("Ctrl+Left Btn - Zoom.\n");
  printf("Shift+Left Btn - Pan.\n");
}

double totalQuery = 0;
bool verb = false; // true;

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: %s model1.obj model2\n", argv[0]);
    return -1;
  }
#ifdef USE_GPU
  printf("Using GPU.\n");
#else
  printf("Using CPU.\n");
#endif

  usage_self_cd();

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_STENCIL);
  glutInitWindowSize(win_w, win_h);
  glutCreateWindow("Collison Detection Tests");

  initOpengl();
  initModel(argv[1], argv[2]);
  // key3();

  object.configure_buttons(1);
  object.dolly.dolly[2] = -3;
  object.trackball.incr = rotationf(vec3f(1, 1, 0), 0.05);

  glutDisplayFunc(display);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutIdleFunc(idle);
  // glutKeyboardFunc(key_self_cd);
  glutKeyboardFunc(key);
  glutReshapeFunc(resize);

  initMenu();

  initSetting();

  glutMainLoop();

  quit();
  return 0;
}

void CaptureScreen(int Width, int Height) {
#ifdef WIN32
  static int captures = 0;
  char filename[20];

  sprintf(filename, "Data/%04d.bmp", captures);
  captures++;

  BITMAPFILEHEADER bf;
  BITMAPINFOHEADER bi;

  char *image = new char[Width * Height * 3];
  FILE *file = fopen(filename, "wb");

  if (image != NULL) {
    if (file != NULL) {
      glReadPixels(0, 0, Width, Height, GL_BGR_EXT, GL_UNSIGNED_BYTE, image);

      memset(&bf, 0, sizeof(bf));
      memset(&bi, 0, sizeof(bi));

      bf.bfType = 'MB';
      bf.bfSize = sizeof(bf) + sizeof(bi) + Width * Height * 3;
      bf.bfOffBits = sizeof(bf) + sizeof(bi);
      bi.biSize = sizeof(bi);
      bi.biWidth = Width;
      bi.biHeight = Height;
      bi.biPlanes = 1;
      bi.biBitCount = 24;
      bi.biSizeImage = Width * Height * 3;

      fwrite(&bf, sizeof(bf), 1, file);
      fwrite(&bi, sizeof(bi), 1, file);
      fwrite(image, sizeof(unsigned char), Height * Width * 3, file);

      fclose(file);
    }
    delete[] image;
  }
#endif
}
