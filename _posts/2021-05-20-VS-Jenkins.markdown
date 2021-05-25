---
layout:     post
title:      "Jenkins"
subtitle:   "Jenkins - Tutorial"
date:       2021-05-20 20:00:00
author:     "Becks"
header-img: "img/post-bg3.jpg"
catalog:    true
tags:
    - Tools
---

如果想open code directiory from command line, 可以用code command

```shell
code Unit-Test-Demo/

```

## Continously integration

Continous Integration Tools (CI): All open source

- Bamboo
- Buildbot
- Gump
- Travis: specialized for Github
- Jenkins


#### Agile

**Advantage**:

1. Client requirements are better understood because of the constant feedback
2. Product is delivered much faster as compared to water fall model. You deliver feature at the end of each sprint(typically two weeks) instead of waiting 6 months


Disadvantage:

1. The products gets tested only on developer computers and not on production systems
2. Developers and operations team work in silos. It is difficult if two teams not work together


#### Before Jenkins

Different developers working at different locations and commit to repository different time. 

1. Issue in integration
2. Delay in Testing. Notify if there are bugs => delay

Developers had to wait till the entire software code was built and tested to check for errors. There was no iterative improvement of code and software delivery process was slow  


#### What is Jenkins

Jenkins is a <span style="color:red">**continuous integration**</span> tool that allows continuous development, test and deployment of newly created codes.

1. Nightly build and integration (Old & legacy approach)
2. Continuous build and Integration: <span style="background-color:#FFFF00">put your test and verification services into the build environment. Always run in cycle to test your code </span>

![](img/post/Jenkins/Jenkins1.png)


#### Continuous Integration

At anytime, able to commit into repo. Submit the code into the **Continous Integration server**. The goal of Continous Integration server is to pass any test that is created. If continous integration server test pass, then that code can sent back to developer. Then developer can make the changes.

It allows developers to do:

1. Allow developer not to break the build 
2. Allow developer not to run all the test locally 


![](img/post/Jenkins/Jenkins2.png)

Running test costs a lot of time, Can put Continuous Integration server into another environment.It improves the productivity of developer

The goal: let release and deploy faster and let customer to get the code faster. When they got code, it works






#### File ICON THEME

1. 到VSCode Extension 下载Ayu 然后Reload
2. 然后Ctrl + Shift + P, Preference:File Icon Theme --> Ayu







## C++ Windows

#### 下载MINGW

1. 下载MinGW Installation Manager
2. click install
3. 右键所有的Package -> mark for installation, 再点Installation ->  ApplyChanges

![](/img/post/VSCode/MinGW1.png)

![](/img/post/VSCode/MinGW2.png)
