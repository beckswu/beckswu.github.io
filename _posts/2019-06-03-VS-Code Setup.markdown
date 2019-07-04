---
layout:     post
title:      "VSCode - Set-up"
subtitle:   "Visual Studio Code 配置"
date:       2019-07-03 20:00:00
author:     "Becks"
header-img: "img/post-bg3.jpg"
catalog:    true
tags:
    - Visual Studio Code
    - VS
    - 配置
    - Setup
---

## Executable

```shell

cmake_minimum_required(VERSION 3.14.5) #VERSION  必须大写，否则报错
project(MyProject VERSION 1.0)

add_executable(cmake-good main.cpp) #make exectuable, 生成exe会叫cmake-good

```

```shell

#build CMakeFiles
cmake .  #引用source directory
 
#build exe
cmake --build . #引用already configured binary directory, 必须要有cache的文件, cache文件会由cmake . 生成 
# windows 环境中 会把exe 建立在Debug folder 中, 因为默认是DEBUG

cmake --build . --config Debug 
#等同于上边的command,
#cmake will run underlying msbuild tool to build VS Solution


```


## Linked a library


比如下面C++ header 和cpp 文件

```c++

//---------------hello.h 

namespace Hello{
    void sayhello();
}

//---------------hello.cpp 

#include "hello.h"

#include <iostream>

void hello::say_hello(){
    coud << " hi "<<endl;
}


//---------------main.cpp 

#include <iostream>

#include "hello.h"

using namespace std;

int main(){
    hello::say_hello();
}

```


如果这么按照上面的方式executable 那么写cmake, run exe时候会有link error， 解决方法是加上add_library 和 target_link_libraries, 


**<span style="background-color:#FFFF00">add_library()</span>** 第一个参数是link library名字(可以随意建立) + library 模式(static or shared), 第二个参数 是头文件，第三个参数 cpp文件

**<span style="background-color:#FFFF00">target_link_libraries()</span>** 第一个参数是executable, link到谁， 第二个参数是link interface mode, 第三个参数 被link的library

<span style="color:red">加上add_library 和 target_link_libraries run cmake --build . (on windows) or make (on linux), 不用run cmake . </span>, 因为会automatically detect 如果CmakeLists.txt 被更改了 (out of date), 如果更改了，会自动run cmake .

```shell
cmake_minimum_required(VERSION 3.14.5) #VERSION  必须大写，否则报错
project(MyProject VERSION 1.0)

add_library( 
    say-hello  #创建一个link library
    hello.h
    hello.cpp
)

add_executable(cmake-good main.cpp) #make exectuable, 生成exe会叫cmake-good

target_link_libraries(cmake-good PRIVATE say-hello) #把建立的library 和main link到一起
#第一个argument是executable, link到谁， 第二个argument是link interface mode, 第三个argument 被link的library


```

再run Debug\cmake-good.exe 会打印出我们想要的结果

注: we don't specify the type of library, but it build a static library(default), 如果我们specified SHARED, 如下面，就会建立一个shared linked library

```shell

add_library( 
    say-hello SHARED  #创建一个shared linked library
    hello.h
    hello.cpp
)

add_library( 
    say-hello STATIC  #创建一个static linked library
    hello.h
    hello.cpp
)



```

在linux 上，ldd 可以看linker dependency of lib,<span style="background-color:#FFFF00"> 如果是shared library, ldd 可以看见libsay-hello (我们上面创建的lib)的dependency, 如果是static, ldd就看不见这个library </span>

```shell
ldd cmake-good

```

我们可以更改add_library static library的选项 成shared, <span style="background-color: #FFFF00">通过cmake -D BUILD_SHARED_LIBS=TRUE . , 即使我们不声明add_library 成shared, 也会生成shared library </span>

```shell

cmake -D BUILD_SHARED_LIBS=TRUE .

```



## CMAKE in Visual Studio

#### MSBUILD

ZERO-CHECK： check if CMAKE configuration file is out of date and needs to regenerate VS studio project. All projects will depend on ZERO CHECK project. You can aslo build ZERO CHECK within Visual Studio itself to regenerate the configuration within visual studio without building any projects

当compile project, can run as executable by right-click set as start project then start debugger.  

当在visual studio 里面更改Debug 到release， 不用重新run cmake 直接build project即可, cmake will run the underlying msbuild build tool in order to build this visual studio solution. Cmake natively understands how to drive MS build to build generated projects. It means you can using Visual studio Solution even without having installation of Visual studio. <span "background-color:#FFFF00"> Visual C++ build tools are sufficient to generate and build Visual Studio Solutions without having the actual IDE installed (Visual C++ 足够generate solution 即使没有装IDE).</span>


 We can also build specific targets by using -- target flag

```shell

cmake --build . --config Debug target simple

```

可以在CMakeLists.txt 文件夹中右键open in visual studio. Visual Studio detects the CMakeLists.txt and open natively without any special commands needing to be run. It show file structure in solution Explore as if it had generated a real solution. Visual Studio Output windows 中可以看见 Visual Studio会自动run cmake for us and generate ninja

#### NINJA

**CL.exe**

Visual C++ 可以是 MSVC or microsoft visual C++ 缩写. 就像GCC and clang, visual C++ 也有 a command line called *CL.EXE* and the linker is called *link.exe*, 如果装了Visual C++, 可能有很多的CL.exe 和 Link.exe, target不同的platform

比如run下面的会报错，因为unlike GCC and Clang,  CL.exe has no built-in default include path. You need to either set in command line or special environment that CL.exe will check and use when it does the search. 最简单的方法set environment is to use Visual Studio Developer Command Prompt, 打开它会显示Environment initialized, <span style="background-color:#FFFF00">因为它include batch file that set the environment variables necessary to compile and link programs</span>, 在Developer Command prompt 就可以run: cl main.cpp (we don't generate any build system, and compile and link to this program manually by executing cl main.cpp )

```shell

#在普通command prompt中run 会报错
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.21.27702\bin\Hostx64\x64\cl.exe" main.cpp 

#在developer command prompt run
cl main.cpp 

```

```shell
#developer command prompt 中
pwsh #切换到powershell

ls Env:  #查看所有的environment variable

echo $$env:INCLUDE$ #可以看developer command prompt的include path
echo $$env:LIB$ #可以看developer command prompt linker search library path

```
![](/img/post/cmake/NINJA.png)

**vcvarshall**

```shell 

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 
#在普通command 上run 这个会切换到与developer command prompt中一样的environment vairable
#CL.exe is on path and ready is use


```


**nmake**

nmake is a partial reimplementation of standard UNIX make, go to cmake project folder (有CMakeLists.txt的), build using nmake. Nmake instead of producing MSBuild file, it will produce nmake format files that can be executed with nmake。 Then ask nmake to build cmake executable for us (与cmake build 花的时间差不多),

NMake 和 CMake 花的时间都挺长的，可以用**NINJA**, 有可能报错if GCC or Clang in your path 因为NINJA called GCC compiler, because NINJA is a make file generaterator 即使ninja doesn't emit make file style build files, 需要specify CMAKE_C_COMPILE 是CL

NINJA 最快，因为during compilation tests, CMake generates a small C make project and tries to build and compile it. Ninja is inherently much faster because it's targeted specifically at compiling large numbers of source files in parallel as much as it can. (Ninja 被用于Chrome project to build webbrowser 因为Chrome 有enormous number of source files)

Using NINJA：Github download NINJA, 需要把NINJA add to system search path 

```shell

#Load Environment Variable
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 

mkdir build
cd build
cmake -G "NMake Makefiles" .. #using nmake as target generator #产生cmake files

nnamke cmake #build cmake executable using name

cmake ..  (与cmake build 花的时间差不多)

#用NINJA Build 
cmake -G Ninja ..

#specify compiler
cmake -G Ninja -DCMAKE_C_COMPILE=cl -DCMAKE_C_COMPILE = cl .. #产生cmake files
ninja cmake  #build project
```

<span style="background-color:#FFFF00">**Summary:  Visual Studio Configure 最慢, NMAKE configure 第二慢, Ninja Configure 最快**</span>


## Visual Studio Code

下载extension cmake (provide by twxs to support syntax highlight ) & cmake tools (it provide extra feature and tweaks helped support cmake based project in editor), 然后点view -> command palette -> 输入 cmcon (Cmake configure) 选择想要的compiler


## Subdirectory &Target

**add_subdirectory** 会include subdirectory的CMakeLists.txt

```c+

/*------------------Hello.h---------------------*/

namespace hello
{
	void say_hello();
};


/*------------------Hello.cpp---------------------*/

#include "hello.h"
#include <iostream>
using namespace std;

void hello::say_hello(){
    cout << " hi "<<endl;
}



/*------------------main.cpp---------------------*/
#include <iostream>
#include "say-hello/hello.h"
using namespace std;

int main(){
    hello::say_hello();
	system("Pause");
}

```


```shell

建立如下面的folder 

say-hello
    -- src
        --say-hello
            --hello.h
            --hello.cpp
    -- CMAKELISTS.txt (1)

hello-exe
   -- CMAKELISTS.txt (2)
   -- main.cpp
   
CMAKELISTS.txt (3)


```


CMAKELISTS.txt (1)
```shell

add_library( 
    say-hello  #创建一个link library
    src/say-hello/hello.h #让path match relative to CMakeLists.txt
    src/say-hello/hello.cpp #让path match relative to CMakeLists.txt
)

#因为main 与 hello.h 不在一个文件夹，所以要把hello.h include 到search path
target_include_directories(say-hello PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
#PUBLIC 是 interface mode
#${CMAKE_CURRENT_SOURCE_DIR} refer to the  directory process 现在这个cmakelists, 就是say-hello 这个directory

```

CMAKELISTS.txt (2)
```shell

add_executable(cmake-good main.cpp) #make exectuable, 生成exe会叫cmake-good

target_link_libraries(cmake-good PRIVATE say-hello) #把建立的library 和main link到一起
#第一个argument是executable, link到谁， 第二个argument是link interface mode, 第三个argument 被link的library

```


CMAKELISTS.txt (3)
```shell

cmake_minimum_required(VERSION 3.14.5) #VERSION  必须大写，否则报错
project(MyProject VERSION 1.0)

add_subdirectory(say-hello) #say-hello 是directory的名字
#会include subdirectory的CMakeLists.txt

add_subdirectory(hello-exe) 
#因为hello-exe中的main call 了 say-hello 中的header 所以需要放在后面
#还以为hello-exe中的CMakeLists.txt的target_link_libraries call 了say-hello library

```


