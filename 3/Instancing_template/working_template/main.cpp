#include <iostream>
#include <vector>
#include <cmath>

#include "Utility.h"
#include "SOIL.h"

using namespace std;

const uint GRASS_INSTANCES = 10000; // Количество травинок

GL::Camera camera;               // Мы предоставляем Вам реализацию камеры. В OpenGL камера - это просто 2 матрицы. Модельно-видовая матрица и матрица проекции. // ###
                                 // Задача этого класса только в том чтобы обработать ввод с клавиатуры и правильно сформировать эти матрицы.
                                 // Вы можете просто пользоваться этим классом для расчёта указанных матриц.


GLuint grassPointsCount; // Количество вершин у модели травинки
GLuint grassShader;      // Шейдер, рисующий траву
GLuint grassVAO;         // VAO для травы (что такое VAO почитайте в доках)
GLuint grassVariance;    // Буфер для смещения координат травинок
vector<VM::vec4> grassVarianceData(GRASS_INSTANCES); // Вектор со смещениями для координат травинок
GLuint groundTex;        // Текстура для земли
GLuint grassTex;         // Текстура для травы

GLuint groundShader; // Шейдер для земли
GLuint groundVAO; // VAO для земли

// Размеры экрана
uint screenWidth = 800;
uint screenHeight = 600;

// Это для захвата мышки. Вам это не потребуется (это не значит, что нужно удалять эту строку)
bool captureMouse = true;

// Функция, рисующая замлю
void DrawGround() {
    // Используем шейдер для земли
    glUseProgram(groundShader);                                                  CHECK_GL_ERRORS

    // Устанавливаем юниформ для шейдера. В данном случае передадим перспективную матрицу камеры
    // Находим локацию юниформа 'camera' в шейдере
    GLint cameraLocation = glGetUniformLocation(groundShader, "camera");         CHECK_GL_ERRORS
    // Устанавливаем юниформ (загружаем на GPU матрицу проекции?)                                                     // ###
    glUniformMatrix4fv(cameraLocation, 1, GL_TRUE, camera.getMatrix().data().data()); CHECK_GL_ERRORS

    // Подключаем VAO, который содержит буферы, необходимые для отрисовки земли
    glBindVertexArray(groundVAO);                                                CHECK_GL_ERRORS

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, groundTex);
    // Рисуем землю: 2 треугольника (6 вершин)
    glDrawArrays(GL_TRIANGLES, 0, 6);                                            CHECK_GL_ERRORS

    glBindTexture(GL_TEXTURE_2D, 0);

    // Отсоединяем VAO
    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    // Отключаем шейдер
    glUseProgram(0);                                                             CHECK_GL_ERRORS
}

//float wind_force = 1;
//float k = 1;
//float m = 0.005;
//vector<float> v{0}; //initial velocity
bool down = true;
int t = 0;

// Обновление смещения травинок
void UpdateGrassVariance() {
    // Генерация случайных смещений
    if (t == 0) {
        down = true;
    } else if (t == 70) {
        down = false;
    }
    if (down) {
        ++t;
    } else {
        --t;
    }
    for (uint i = 0; i < GRASS_INSTANCES; ++i) {
        if (down) {
            grassVarianceData[i].x -= 0.0001;
            grassVarianceData[i].y -= 0.0001;
        } else {
            grassVarianceData[i].x += 0.0001;
            grassVarianceData[i].y += 0.0001;
        }
        //float a = (wind_force - k * grassPositions[i].x) / m;
        //v[i] = v[i] + a * t;
        //grassPositions[i].x = grassPositions[i].x + v[i] * t;
        // if (grassVarianceData[i].x > 0.05) {
        //     neg = true;
        // } else if (grassVarianceData[i].x <= 0) {
        //     neg = false;
        // }
        // if (neg) {
        //     grassVarianceData[i].x -= 0.001;
        // } else {
        //     grassVarianceData[i].x += 0.001;
        // }
        //grassVarianceData[i].z = (float)rand() / RAND_MAX / 100;
        continue;
    }

    // Привязываем буфер, содержащий смещения
    glBindBuffer(GL_ARRAY_BUFFER, grassVariance);                                CHECK_GL_ERRORS
    // Загружаем данные в видеопамять
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * GRASS_INSTANCES, grassVarianceData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS
    // Отвязываем буфер
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS
}


// Рисование травы
void DrawGrass() {
    // Тут то же самое, что и в рисовании земли
    glUseProgram(grassShader);                                                   CHECK_GL_ERRORS
    GLint cameraLocation = glGetUniformLocation(grassShader, "camera");          CHECK_GL_ERRORS
    glUniformMatrix4fv(cameraLocation, 1, GL_TRUE, camera.getMatrix().data().data()); CHECK_GL_ERRORS
    glBindVertexArray(grassVAO);                                                 CHECK_GL_ERRORS
    // Обновляем смещения для травы
    UpdateGrassVariance();
    // Отрисовка травинок в количестве GRASS_INSTANCES
    glBindTexture(GL_TEXTURE_2D, grassTex);                                     CHECK_GL_ERRORS

    glDrawArraysInstanced(GL_TRIANGLES, 0, grassPointsCount, GRASS_INSTANCES);   CHECK_GL_ERRORS

    glBindTexture(GL_TEXTURE_2D, 0);                                            CHECK_GL_ERRORS

    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    glUseProgram(0);                                                             CHECK_GL_ERRORS
}

// Эта функция вызывается для обновления экрана
void RenderLayouts() {
    // Включение буфера глубины
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    // Очистка буфера глубины и цветового буфера
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Рисуем меши
    DrawGround();
    DrawGrass();
    glutSwapBuffers();
}

// Завершение программы
void FinishProgram() {
    glutDestroyWindow(glutGetWindow());
}

// Обработка события нажатия клавиши (специальные клавиши обрабатываются в функции SpecialButtons)
void KeyboardEvents(unsigned char key, int x, int y) {
    if (key == 27) {
        FinishProgram();
    } else if (key == 'w') {
        camera.goForward();
    } else if (key == 's') {
        camera.goBack();
    } else if (key == 'm') {
        captureMouse = !captureMouse;
        if (captureMouse) {
            glutWarpPointer(screenWidth / 2, screenHeight / 2);
            glutSetCursor(GLUT_CURSOR_NONE);
        } else {
            glutSetCursor(GLUT_CURSOR_RIGHT_ARROW);
        }
    }
}

// Обработка события нажатия специальных клавиш
void SpecialButtons(int key, int x, int y) {
    if (key == GLUT_KEY_RIGHT) {
        camera.rotateY(0.02);
    } else if (key == GLUT_KEY_LEFT) {
        camera.rotateY(-0.02);
    } else if (key == GLUT_KEY_UP) {
        camera.rotateTop(-0.02);
    } else if (key == GLUT_KEY_DOWN) {
        camera.rotateTop(0.02);
    }
}

void IdleFunc() {
    glutPostRedisplay();
}

// Обработка события движения мыши
void MouseMove(int x, int y) {
    if (captureMouse) {
        int centerX = screenWidth / 2,
            centerY = screenHeight / 2;
        if (x != centerX || y != centerY) {
            camera.rotateY((x - centerX) / 1000.0f);
            camera.rotateTop((y - centerY) / 1000.0f);
            glutWarpPointer(centerX, centerY);
        }
    }
}

// Обработка нажатия кнопки мыши
void MouseClick(int button, int state, int x, int y) {
}

// Событие изменение размера окна
void windowReshapeFunc(GLint newWidth, GLint newHeight) {
    glViewport(0, 0, newWidth, newHeight);
    screenWidth = newWidth;
    screenHeight = newHeight;

    camera.screenRatio = (float)screenWidth / screenHeight;
}

// Инициализация окна
void InitializeGLUT(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitContextVersion(3, 0);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitWindowPosition(-1, -1);
    glutInitWindowSize(screenWidth, screenHeight);
    glutCreateWindow("WEED");
    glutWarpPointer(400, 300);
    glutSetCursor(GLUT_CURSOR_NONE);
    glClearColor(0.678, 0.847, 0.902, 1);


    glutDisplayFunc(RenderLayouts);
    glutKeyboardFunc(KeyboardEvents);
    glutSpecialFunc(SpecialButtons);
    glutIdleFunc(IdleFunc);
    glutPassiveMotionFunc(MouseMove);
    glutMouseFunc(MouseClick);
    glutReshapeFunc(windowReshapeFunc);
}

// Генерация позиций травинок (эту функцию вам придётся переписать)
vector<VM::vec2> GenerateGrassPositions() {
    vector<VM::vec2> grassPositions(GRASS_INSTANCES);
    for (uint i = 0; i < GRASS_INSTANCES; ++i) {
        //grassPositions[i] = VM::vec2((i % 4) / 4.0, (i / 4) / 4.0) + VM::vec2(1, 1) / 8;
        //grassPositions[i] = VM::vec2((i % uint(sqrt(GRASS_INSTANCES))) / sqrt(GRASS_INSTANCES), (i / uint(sqrt(GRASS_INSTANCES))) / sqrt(GRASS_INSTANCES)) + VM::vec2(1, 1) / (2 * sqrt(GRASS_INSTANCES));
        grassPositions[i] = VM::vec2(0.005 + (float)rand()/RAND_MAX * 0.99, 0.005 + (float)rand()/RAND_MAX * 0.99);
    }
    return grassPositions;
}

// Здесь вам нужно будет генерировать меш
vector<VM::vec4> GenMesh(uint n) {
    return {
/*        //012
        VM::vec4(0, 0, 0, 1), //0
        VM::vec4(1, 0, 0, 1), //1
        VM::vec4(0.5, 1, 0, 1), //2
        //123
        VM::vec4(1, 0, 0, 1), //1
        VM::vec4(0.5, 1, 0, 1), //2
        VM::vec4(1.5, 1, 0, 1), //3
        //234
        VM::vec4(0.5, 1, 0, 1), //2
        VM::vec4(1.5, 1, 0, 1), //3
        VM::vec4(1.5, 2, 0, 1), //4
        //345
        VM::vec4(1.5, 1, 0, 1), //3
        VM::vec4(1.5, 2, 0, 1), //4
        VM::vec4(2.5, 2, 0, 1), //5
        //456
        VM::vec4(1.5, 2, 0, 1), //4
        VM::vec4(2.5, 2, 0, 1), //5
        VM::vec4(3.5, 2.5, 0, 1), //6
*/
/*        //012
        VM::vec4(0, 0, 0, 1), //0
        VM::vec4(1, 0, 0, 1), //1
        VM::vec4(0.5, 1, 0, 1), //2
        //123
        VM::vec4(1, 0, 0, 1), //1
        VM::vec4(0.5, 1, 0, 1), //2
        VM::vec4(1, 1, 0, 1), //3
        //234
        VM::vec4(0.5, 1, 0, 1), //2
        VM::vec4(1, 1, 0, 1), //3
        VM::vec4(1, 5.0/3, 0, 1), //4
        //245
        VM::vec4(0.5, 1, 0, 1), //2
        VM::vec4(1, 5.0/3, 0, 1), //4
        VM::vec4(3.0/4, 2, 0, 1), //5
        //456
        VM::vec4(1, 5.0/3, 0, 1), //4
        VM::vec4(3.0/4, 2, 0, 1), //5
        VM::vec4(1, 8.0/3, 0, 1), //6
*/
        //012
        VM::vec4(0, 0, 0, 1), //0
        VM::vec4(1, 0, 0, 1), //1
        VM::vec4(4.375 / 5, 7.5 / 30, -0.003, 1), //2
        //023
        VM::vec4(0, 0, 0, 1), //0
        VM::vec4(4.375 / 5, 7.5 / 30, -0.003, 1), //2
        VM::vec4(0.625 / 5, 7.5 / 30, -0.003, 1), //3
        //234
        VM::vec4(4.375 / 5, 7.5 / 30, -0.003, 1), //2
        VM::vec4(0.625 / 5, 7.5 / 30, -0.003, 1), //3
        VM::vec4(1.25 / 5, 15.0 / 30, -0.008, 1), //4
        //245
        VM::vec4(4.375 / 5, 7.5 / 30, -0.003, 1), //2
        VM::vec4(1.25 / 5, 15.0 / 30, -0.008, 1), //4
        VM::vec4(3.75 / 5, 15.0 / 30, -0.008, 1), //5
        //456
        VM::vec4(1.25 / 5, 15.0 / 30, -0.008, 1), //4
        VM::vec4(3.75 / 5, 15.0 / 30, -0.008, 1), //5
        VM::vec4(3.125 / 5, 22.5 / 30, -0.013, 1), //6
        //467
        VM::vec4(1.25 / 5, 15.0 / 30, -0.008, 1), //4
        VM::vec4(3.125 / 5, 22.5 / 30, -0.013, 1), //6
        VM::vec4(1.875 / 5, 22.5 / 30, -0.013, 1), //7
        //678
        VM::vec4(3.125 / 5, 22.5 / 30, -0.013, 1), //6
        VM::vec4(1.875 / 5, 22.5 / 30, -0.013, 1), //7
        VM::vec4(2.5 / 5, 26.0 / 30, -0.018, 1), //8

    };
}

// Создание травы
void CreateGrass() {
    uint LOD = 1;
    // Создаём меш
    vector<VM::vec4> grassPoints = GenMesh(LOD);
    // Сохраняем количество вершин в меше травы
    grassPointsCount = grassPoints.size();
    // Создаём позиции для травинок
    vector<VM::vec2> grassPositions = GenerateGrassPositions();
    // Инициализация смещений для травинок
    for (uint i = 0; i < GRASS_INSTANCES; ++i) {
        grassVarianceData[i] = VM::vec4(0.004 * (float)rand() / RAND_MAX / 10, 0, 0, 0);
    }

    /* Компилируем шейдеры
    Эта функция принимает на вход название шейдера 'shaderName',
    читает файлы shaders/{shaderName}.vert - вершинный шейдер
    и shaders/{shaderName}.frag - фрагментный шейдер,
    компилирует их и линкует.
    */
    grassShader = GL::CompileShaderProgram("grass");

    // Здесь создаём буфер
    GLuint pointsBuffer;
    // Это генерация одного буфера (в pointsBuffer хранится идентификатор буфера)
    glGenBuffers(1, &pointsBuffer);                                              CHECK_GL_ERRORS
    // Привязываем сгенерированный буфер
    glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);                                 CHECK_GL_ERRORS
    // Заполняем буфер данными из вектора
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * grassPoints.size(), grassPoints.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    // Создание VAO
    // Генерация VAO
    glGenVertexArrays(1, &grassVAO);                                             CHECK_GL_ERRORS
    // Привязка VAO
    glBindVertexArray(grassVAO);                                                 CHECK_GL_ERRORS

    // Получение локации параметра 'point' в шейдере
    GLuint pointsLocation = glGetAttribLocation(grassShader, "point");           CHECK_GL_ERRORS
    // Подключаем массив атрибутов к данной локации
    glEnableVertexAttribArray(pointsLocation);                                   CHECK_GL_ERRORS
    // Устанавливаем параметры для получения данных из массива (по 4 значение типа float на одну вершину)
    glVertexAttribPointer(pointsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);          CHECK_GL_ERRORS

    // Создаём буфер для позиций травинок
    GLuint positionBuffer;
    glGenBuffers(1, &positionBuffer);                                            CHECK_GL_ERRORS
    // Здесь мы привязываем новый буфер, так что дальше вся работа будет с ним до следующего вызова glBindBuffer
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec2) * grassPositions.size(), grassPositions.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    GLuint positionLocation = glGetAttribLocation(grassShader, "position");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(positionLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    // Здесь мы указываем, что нужно брать новое значение из этого буфера для каждого инстанса (для каждой травинки)
    glVertexAttribDivisor(positionLocation, 1);                                  CHECK_GL_ERRORS

    // Создаём буфер для смещения травинок
    glGenBuffers(1, &grassVariance);                                            CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, grassVariance);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * GRASS_INSTANCES, grassVarianceData.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    GLuint varianceLocation = glGetAttribLocation(grassShader, "variance");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(varianceLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(varianceLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    glVertexAttribDivisor(varianceLocation, 1);                                  CHECK_GL_ERRORS

    // Создаем вектор для размеров травинок
    vector<VM::vec2> grassScales;
    for (uint i = 0; i < grassPositions.size(); ++i) {
        float width_scale = 0.002 + (float)rand() / RAND_MAX * 0.004;
        float height_scale = 0.05 +(-1 + (float)rand() / RAND_MAX * 2) * 0.04;
        grassScales.push_back(VM::vec2(width_scale, height_scale));
    }

    // Создаём буфер для размеров травинок
    GLuint scaleBuffer;
    glGenBuffers(1, &scaleBuffer);                                            CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, scaleBuffer);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec2) * grassPositions.size(), grassScales.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    GLuint scalesLocation = glGetAttribLocation(grassShader, "scales");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(scalesLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(scalesLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    glVertexAttribDivisor(scalesLocation, 1);                                  CHECK_GL_ERRORS


    // Создаем вектор углов поворота травинок
    vector<VM::vec2> grassAngles;
    for (uint i = 0; i < grassPositions.size(); ++i) {
        float angle = (float)rand() / RAND_MAX * 2 * M_PI;
        grassAngles.push_back(VM::vec2(cos(angle), sin(angle)));
    }

    // Создаём буфер для углов поворота травинок
    GLuint angleBuffer;
    glGenBuffers(1, &angleBuffer);                                            CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, angleBuffer);                               CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec2) * grassPositions.size(), grassAngles.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    GLuint anglesLocation = glGetAttribLocation(grassShader, "y_cos_sin");      CHECK_GL_ERRORS
    glEnableVertexAttribArray(anglesLocation);                                 CHECK_GL_ERRORS
    glVertexAttribPointer(anglesLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    glVertexAttribDivisor(anglesLocation, 1);                                  CHECK_GL_ERRORS

    // Создаем вектор цветов
    // vector<VM::vec4> grassColors;
    // for (uint i = 0; i < grassPositions.size(); ++i) {
    //     float shift1 = (-5 + (float)rand() / RAND_MAX * 10 ) / 255;
    //     float shift2 = (-10 + (float)rand() / RAND_MAX * 40 ) / 255;
    //     float shift3 = (-5 + (float)rand() / RAND_MAX * 10 ) / 255;
    //     grassColors.push_back(VM::vec4(0.180 + shift1, 0.545 + shift2, 0.341 + shift3, 0));
    // }
    //
    //Создаём буфер для цветов травинок
    // GLuint colorBuffer;
    // glGenBuffers(1, &colorBuffer);                                            CHECK_GL_ERRORS
    // glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);                               CHECK_GL_ERRORS
    // glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * grassPositions.size(), grassColors.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS
    //
    // GLuint colorsLocation = glGetAttribLocation(grassShader, "in_color");      CHECK_GL_ERRORS
    // glEnableVertexAttribArray(colorsLocation);                                 CHECK_GL_ERRORS
    // glVertexAttribPointer(colorsLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);        CHECK_GL_ERRORS
    // glVertexAttribDivisor(colorsLocation, 1);                                  CHECK_GL_ERRORS


    // Отвязываем VAO
    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    // Отвязываем буфер
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS

    glGenTextures(1, &grassTex);                                                CHECK_GL_ERRORS
    glBindTexture(GL_TEXTURE_2D, grassTex);                                     CHECK_GL_ERRORS

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height;
    unsigned char* image = SOIL_load_image("Texture/grass.jpg", &width, &height, 0, SOIL_LOAD_RGBA); CHECK_GL_ERRORS
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
              GL_UNSIGNED_BYTE, image);                                          CHECK_GL_ERRORS
    glGenerateMipmap(GL_TEXTURE_2D);                                             CHECK_GL_ERRORS
    SOIL_free_image_data(image);                                                CHECK_GL_ERRORS
    glBindTexture(GL_TEXTURE_2D, 0);                                             CHECK_GL_ERRORS
    int a = 0;
    if (a) {
        cout << "wtf\n";
    }
}

// Создаём камеру (Если шаблонная камера вам не нравится, то можете переделать, но я бы не стал)
void CreateCamera() {
    camera.angle = 45.0f / 180.0f * M_PI;
    camera.direction = VM::vec3(0, 0.3, -1);
    camera.position = VM::vec3(0.5, 0.2, 0);
    camera.screenRatio = (float)screenWidth / screenHeight;
    camera.up = VM::vec3(0, 1, 0);
    camera.zfar = 50.0f;
    camera.znear = 0.05f;
}

// Создаём замлю
void CreateGround() {
    // Земля состоит из двух треугольников
    vector<VM::vec4> meshPoints = {
        VM::vec4(0, 0, 0, 1),
        VM::vec4(1, 0, 0, 1),
        VM::vec4(1, 0, 1, 1),
        VM::vec4(0, 0, 0, 1),
        VM::vec4(1, 0, 1, 1),
        VM::vec4(0, 0, 1, 1),
    };

    // Подробнее о том, как это работает читайте в функции CreateGrass

    groundShader = GL::CompileShaderProgram("ground");

    GLuint pointsBuffer;
    glGenBuffers(1, &pointsBuffer);                                              CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);                                 CHECK_GL_ERRORS
    glBufferData(GL_ARRAY_BUFFER, sizeof(VM::vec4) * meshPoints.size(), meshPoints.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS

    glGenVertexArrays(1, &groundVAO);                                            CHECK_GL_ERRORS
    glBindVertexArray(groundVAO);                                                CHECK_GL_ERRORS

    GLuint index = glGetAttribLocation(groundShader, "point");                   CHECK_GL_ERRORS
    glEnableVertexAttribArray(index);                                            CHECK_GL_ERRORS
    glVertexAttribPointer(index, 4, GL_FLOAT, GL_FALSE, 0, 0);                   CHECK_GL_ERRORS

    glBindVertexArray(0);                                                        CHECK_GL_ERRORS
    glBindBuffer(GL_ARRAY_BUFFER, 0);                                            CHECK_GL_ERRORS


    glGenTextures(1, &groundTex);
    glBindTexture(GL_TEXTURE_2D, groundTex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height;
    unsigned char* image = SOIL_load_image("Texture/zeml2.png", &width, &height, 0, SOIL_LOAD_RGBA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
              GL_UNSIGNED_BYTE, image);
    glGenerateMipmap(GL_TEXTURE_2D);
    SOIL_free_image_data(image);
    glBindTexture(GL_TEXTURE_2D, 0);

    //glGetUniformLocation(groundShader, "tex");
}

int main(int argc, char **argv)
{
    try {
        cout << "Start" << endl;
        InitializeGLUT(argc, argv);
        cout << "GLUT inited" << endl;
        glewInit();
        cout << "glew inited" << endl;
        CreateCamera();
        cout << "Camera created" << endl;
        CreateGrass();
        cout << "Grass created" << endl;
        CreateGround();
        cout << "Ground created" << endl;
        glutMainLoop();
    } catch (string s) {
        cout << s << endl;
    }
}
