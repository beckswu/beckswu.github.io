---
layout:     post
title:      "Vcpkg Installation"
subtitle:   "vcpkg installation on Windows"
date:       2017-09-28 10:00:00
author:     "Becks"
header-img: "img/post/math.jpg"
catalog:    false
tags: 
    - C++ library
---

> quick guide to install vcpkg
> 



## Auto Linked Library for C++ Visual Studio



__1__. Install, [vcpkg](https://github.com/Microsoft/vcpkg) 
Open Github cmd

```shell
> git clone https://github.com/Microsoft/vcpkg
> cd vcpkg
``` 

__2__. Then, to hook up user-wide integration(让以后的library都可以通过vcpkg安装), run (note: requires admin on first use), Use PowerShell

```shell
PS> .\vcpkg integrate install

```

__3__. Install library via vcpkg

```shell
> cd vcpkg
PS> .\vcpkg install cpprestsdk cpprestsdk:x64-windows
PS> .\vcpkg install boost:x86-windows
```

