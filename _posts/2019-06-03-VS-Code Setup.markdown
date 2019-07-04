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

## C++ Windows

#### 下载MINGW

1. 下载MinGW Installation Manager
2. click install
3. 右键所有的Package -> mark for installation, 再点Installation ->  ApplyChanges

![](/img/post/VSCode/MinGW1.png)

![](/img/post/VSCode/MinGW2.png)


#### add MINGW to system path

1. 控制面板--> 系统和安全 --> 系统  --> 高级系统设置
2. 高级 --> 环境变量
3. 系统变量内找到Path, 双击编辑, 把;C:\MinGW\bin 加在变量值的后面

![](/img/post/VSCode/MinGW3.png)

![](/img/post/VSCode/MinGW4.png)

![](/img/post/VSCode/MinGW5.png)

#### Build C++

1. 建一个folder, folder 里面有main.cpp,  需要build的程序
2. Ctrl + shift + p , search tasks, 选择 Tasks: Configure Task -->  选择create task.json file from template --> 选择 Others (Example to run an arbitrary external command）, 会生成一个tasks.json文件， 我们把 "command" 改成 "g++ -g main.cpp"   (-g is debug) or 写成如下图格式
3. Ctrl + Shift + B 可以build project (如果出现permission defined 错误，关掉VS Code, 重新打开再Ctrl+Shift +B, build)
4. 可以在VS Code中新建terminal, 点plus button. run  a.exe

![](/img/post/VSCode/task0.png)

![](/img/post/VSCode/task1.png)

![](/img/post/VSCode/task2.png)

![](/img/post/VSCode/task3.png)

注：不用Ctrl + shift + p, search C\Cpp: Edit Configurations, 在新的VS Code的configuration中自动设置好了MinGW include path



查看是不是把 MinGW 放进Environment variable,打开prompt, 输入 

```shell

g++ --version #如果MinGW 放进Environment variable，会显示信息

```

#### Debug in VS Code

1. 点击左侧 Debug Panel, 点击绿色箭头 --> 因为我们用MinGW, 选择C++ (GDB//LLDB), 生成一个Default lauch.json 
2. 修改launch.json
	- 修改 "miDebuggerPath": "C:\\MinGW\\bin\\gdb.exe"  (gdb.exe 是debugger)
	- add "preLaunchTask": "echo",  这个preLaunchTask 需要与tasks.json task 命名一样, (是为了build code first, then start debugging)
	- 修改"program": "${workspaceFolder}/a.exe", 这是告诉什么exe 用来debug的
3. 可以再点击绿箭头, 开始Debug, 可以设置Break point,来方便debug
	- 当开始Debug,可以控制到下一个step over break point,step into, step out

![](/img/post/VSCode/win_Debug1.png)

![](/img/post/VSCode/win_Debug2.png)

注: stopAtEntry, 表示当debug开始, 程序一开始先停住

![](/img/post/VSCode/win_Debug3.png)



## C++ Mac

#### 下载code runner

1. 到extension 中找到code runner，下载
2. 到user/用户名/⁨ .vscode⁩ / ⁨extensions⁩ / ⁨formulahendry.code-runner-0.9.10⁩ / out⁩ /,codeManager.js, comment 掉line 12 和 line 225~236, 如图
3. 重新加载，或关掉VS Code 再打开，点击右上角箭头，run, 但这时候如果code 有cin, 不能在输出panel上 输入到程序


![](/img/post/VSCode/coderunner1.png)

![](/img/post/VSCode/coderunner2.png)

![](/img/post/VSCode/coderunner3.png)

4. 点右下角 -->  设置, 在用户设置中 (user settings) 加上 "code-runner.runInTerminal": true
5. 再点run code时候，就会在terminal 中run code, 

![](/img/post/VSCode/coderunner4.png)

![](/img/post/VSCode/coderunner5.png)


#### 设置C++ 版本

1. 点右下角 -->  设置, 在用户设置中 (user settings) 输入 ”code-runner.executorMap":, 然后点tooltip，会蹦出default的格式,
2. 只保留cpp, 其他的可以删掉，在g++ 后面加上 -std=c++17, 再点run code 会显示c++版本

![](/img/post/VSCode/version1.png)

![](/img/post/VSCode/version2.png)


#### Debug

1. Ctrl + Shift + P, Tasks,--> Tasks: Configure Task --> create task.json file from template -->  Others
2. change command and add args (如下图), 设置后之后, Ctrl + Shift + B 就可以build code了


![](/img/post/VSCode/mac_Debug1.png)

3. 修改launch.json 如下图, ( preLaunchTask 和 program) , 之后就可以设置break point Debug了

![](/img/post/VSCode/mac_Debug2.png)