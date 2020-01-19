---
layout:     post
title:      "Bash Script - 笔记"
subtitle:   "Bash Script learning note "
date:       2019-06-01 22:00:00
author:     "Becks"
header-img: "img/post-bg2.jpg"
catalog:    true
tags:
    - Bash Script
    - 学习笔记
---


## Bash 介绍

```shell
vim hello.sh
```

```#!``` 是说明 hello 这个文件的类型，有点类似于 Windows 系统下用不同文件后缀来表示不同文件类型的意思（但不相同）。
Linux 系统根据 ```#!``` 及该字符串后面的信息确定该文件的类型，可以通过 man magic命令 及 /usr/share/magic 文件来了解这方面的更多内容。 ```#!/bin/bash```这一行是表示使用```/bin/bash```作为脚本的解释器，这行要放在脚本的行首并且<span style="color:red">**不要省略**</span>

在 BASH 中 第一行的 ```#!``` 及后面的 ```/bin/bash``` 就表明该文件是一个 <span style="color:red"> BASH 程序，需要由 /bin 目录下的 bash 程序来解释执行</span>。BASH 这个程序一般是存放在 ```/bin``` 目录下，如果你的 Linux 系统比较特别，bash 也有可能被存放在 ```/sbin``` 、```/usr/local/bin``` 、```/usr/bin``` 、```/usr/sbin``` 或 ```/usr/local/sbin``` 这样的目录下；如果还找不到，你可以用 ```locate bash``` ,```find / -name bash 2>/dev/null``` 或 ```whereis bash``` 这三个命令找出 bash 所在的位置；如果仍然找不到，那你可能需要自己动手安装一个 BASH 软件包了。

第二行的 ```# This is a ...``` 就是 BASH 程序的注释，在 BASH 程序中从```#```号（注意：后面紧接着是```!```号的除外）开始到行尾的部分均被看作是程序的注释。

需要注意的是 BASH 中的绝大多数语句结尾处都没有分号。


#### 运行Bash脚本的方式：

```bash
# 使用shell来执行
$ sh hello.sh

# 使用bash来执行
$ bash hello.sh

使用.来执行
$ . ./hello.sh

使用source来执行
$ source hello.sh

还可以赋予脚本所有者执行权限，允许该用户执行该脚本
$ chmod u+rx hello.sh
$  ./hello.sh
```


使用脚本清除```/var/log```下的log文件
首先我们看一看```/var/log/wtmp```里面有啥东西
```bash
cat /var/log/wtmp
```
这个文件中记录了系统的一些信息，现在我们需要写一个脚本把里面的东西清空，但是保留文件

```bash
$ vim cleanlogs.sh
```

说明: ```/dev/null```这个东西可以理解为一个黑洞，里面是空的（可以用cat命令看一看）

```bash
#!/bin/bash

# 初始化一个变量
LOG_DIR=/var/log

cd $LOG_DIR

cat /dev/null > wtmp

echo "Logs cleaned up."

exit
```
运行脚本前，先使用 ```sudo chmod +x cleanlogs.sh``` 授予脚本执行权限，然后再看看 ```/var/log/wtmp``` 文件内是否有内容。运行此脚本后，文件的内容将被清除。

执行

由于脚本中<span style="color:red">含有对系统日志文件内容的清除操作，这要求要有管理员权限.不然会报permission denied错误, 使用sudo命令调用管理员权限才能执行成功</span>：
```bash
$ sudo ./cleanlogs.sh
````


脚本正文中以```#```号开头的行都是注释语句，这些行在脚本的实际执行过程中不会被执行。这些注释语句能方便我们在脚本中做一些注释或标记，让脚本更具可读性。

可是你会发现 ```sudo cat /dev/null > /var/log/wtmp``` 一样会提示权限不够，为什么呢？因为```sudo```只能让```cat```命令以```sudo```的权限执行，而对于```>```这个符号并没有sudo的权限，我们可以使用

```bash
sudo sh -c "cat /dev/null > /var/log/wtmp " 让整个命令都具有sudo的权限执行
```

Q: 为什么cleanlogs.sh可以将log文件清除？ <br/>
A: 因为```/dev/null``` ，里面是空的，重定向到 ```/var/log/wtmp``` 文件后，就清空了 wtmp 文件的内容。



#### 特殊字符


| 字符 | 解释 |
| :---: | :--- |
| ```#``` | ```#``` 开头(除```#!```之外)的是注释。```#!```是用于指定当前脚本的解释器，我们这里为bash，且应该指明完整路径，所以为```/bin/bash```, ```\#``` 就不是注释 |
| ```;``` | 1. 使用分号（;）可以在同一行上写两个或两个以上的命令 <br/> 2. 终止case选项（双分号） |
| ```.``` | 点号等价于 source 命令: 当前 bash 环境下读取并执行 FileName.sh 中的命令 |
|```"``` | 双引号, "STRING" 将会阻止（解释）STRING中大部分特殊的字符。见下面例子 |
| ```'``` | 单引号, STRING' 将会阻止STRING中<span style="color:red">**所有特殊字符**</span>的解释 | 
| ```/``` | 1. 斜线（/） 文件名路径分隔符。分隔文件名不同的部分（如/```home/bozo/projects/Makefile```）注意在linux中表示路径的时候，许多个```/```跟一个```/```是一样的。```/home/shiyanlou```等同于```////home///shiyanlou```<br> 2. 也可用来作为除法算术操作符。| 
| ```\``` |  一种对单字符的引用机制。```\X``` 将会“转义”字符```X```。这等价于```"X"```，也等价于```'X'```。```\``` 通常用来转义双引号（```"```）和单引号（```'```），这样双引号和单引号就不会被解释成特殊含义了。 <br/> - ```\n``` 表示新的一行<br/> - ```\r``` 表示回车<br/> -```\t``` 表示水平制表符<br/> -```\v``` 表示垂直制表符<br/> -```\b``` 表示后退符<br/> -```\a``` 表示"alert"(蜂鸣或者闪烁)<br/> -```\0xx``` 转换为八进制的ASCII码, 等价于0xx<br/> -```"``` 表示引号字面的意思 | 


**#**

```shell
#!/bin/bash

echo "The # here does not begin a comment."
echo 'The # here does not begin a comment.'
echo The \# here does not begin a comment.
echo The # 这里开始一个注释
echo $(( 2#101011 ))     # 数制转换（使用二进制表示），不是一个注释，双括号表示对于数字的处理

```

**;**

e.g.1 使用分号（;）可以在同一行上写两个或两个以上的命令

```bash
 #!/bin/bash
 echo hello; echo there
 filename=ttt.sh
 if [ -e "$filename" ]; then    # 注意: "if"和"then"需要分隔，-e用于判断文件是否存在
     echo "File $filename exists."; cp $filename $filename.bak
 else
     echo "File $filename not found."; touch $filename
 fi; echo "File test complete."
 ```
 
 e.g. 2.终止case选项（双分号）

```shelll
$ vim test3.sh
```
输入如下代码，并保存。
```
#!/bin/bash

varname=b

case "$varname" in
    [a-z]) echo "abc";;
    [0-9]) echo "123";;
esac
```
执行脚本，查看输出
```
$ bash test3.sh
abc
```

上面脚本使用case语句，首先创建了一个变量初始化为b,然后使用case语句判断该变量的范围，并打印相关信息。
 
 
 **.**
 
 ```
 $ source test.sh
Hello World
$ . test.sh
Hello World
```

**引号**

![](/img/post/shell/quote.png)

**反引号**

反引号中的命令会优先执行，如：

```
$ cp `mkdir back` test.sh back
$ ls
```

<span style="background-color:#FFFF00">先创建了 back 目录，然后复制 test.sh 到 back 目录</span>



**:冒号**

1. 与true 作用相同

```bash
#!/bin/bash

while :
do
    echo "endless loop"
done
```

```bash

#!/bin/bash

while true
do
    echo "endless loop"
done
```


可以在 if/then 中作占位符：

```bash
#!/bin/bash

condition=5

if [ $condition -gt 0 ] #gt表示greater than，也就是大于，同样有-lt（小于），-eq（等于） 
then :   # 什么都不做，退出分支
else
    echo "$condition"
fi
```

2.变量扩展/子串替换

在与 ```>``` 重定向操作符结合使用时，将会把一个文件清空，但是并不会修改这个文件的权限。如果之前这个文件并不存在，那么就创建这个文件。

```bash
$ : > test.sh   # 文件“test.sh”现在被清空了
# 与 cat /dev/null > test.sh 的作用相同
# 然而,这并不会产生一个新的进程, 因为“:”是一个内建命令
```

在与 ```>>```重定向操作符结合使用时，将不会对预先存在的目标文件 ( ```: >> target_file```)产生任何影响。<span style="background-color:#FFFF00">**如果这个文件之前并不存在，那么就创建它**</span>。

也可能用来作为注释行，但不推荐这么做。使用 ```#``` 来注释的话，将关闭剩余行的错误检查，所以可以在注释行中写任何东西。然而，使用 ```:``` 的话将不会这样。如：

```bash
$ : This is a comment that generates an error,( if [ $x -eq 3] )
```

":"还用来在 /etc/passwd 和 $PATH 变量中做分隔符，如：

```bash
$ echo $PATH
/usr/local/bin:/bin:/usr/bin:/usr/X11R6/bin:/sbi
```


**?问号**

在一个双括号结构中，? 就是C语言的三元操作符，如：


```bash
$ vim test.sh
```

输入如下代码，并保存：

```bash
 #!/bin/bash

 a=10
 (( t=a<50?8:9 ))
 echo $t
```
运行测试
```bash
$ bash test.sh
```


**$美元符号**

前面已经用到了

```bash
$ vim test.sh
```

```bash
#!/bin/bash

var1=5
var2=23skidoo

echo $var1     # 5
echo $var2     # 23skidoo
```

运行测试

```bash
$ bash test.sh
5
23skidoo
```



















#### 特殊符号

| 符号 | 解释 |
| :---: | :--- |
| ```()``` | 1. 在括号中的命令列表，将会作为一个子 shell 来运行, 括号中的变量,对于脚本剩下的部分<span style="background-color:#FFFF00">**是不可用的**</span>. <br/> 2. 初始化数组   |
| ```{}``` | 1. 文件名扩展 比如 ```touch {1..10}.txt``` <br/> 2. 创建了一个匿名函数, <span style="background-color:#FFFF00">**对于大括号外可见**</span> |
| ```[]``` | 1. if condition<br/> 2. 数组index|
| ```<  >``` | 重新定向 |
| ```|``` | 管道 | 
| ```-``` |  破折号: 在所有的命令内如果想使用选项参数的话的前缀. 2.用于重定向stdin或stdout  |
| ```~``` | 波浪号: 表示Home |

**1. 小括号()**

1.命令组

在括号中的命令列表，将会作为一个子 shell 来运行。

在括号中的变量，由于是在子shell中，所以<span style="background-color:#FFFF00">**对于脚本剩下的部分是不可用的**</span>。父进程，也就是脚本本身，将不能够读取在子进程中创建的变量，也就是在子shell 中创建的变量。如：

```bash
$ vim test20.sh
```
输入代码:

```bash
#!/bin/bash

a=123
( a=321; ) #没有分号也可以

echo "$a" #a的值为123而不是321，因为括号将判断为局部变量
```
运行代码：

```
$ bash test20.sh
a = 123
```

在圆括号中 a 变量，更像是一个局部变量。

2.初始化数组

创建数组

```bash
$ vim test21.sh
```

输入代码：

```bash
#!/bin/bash

arr=(1 4 5 7 9 21)
echo ${arr[3]} # get a value of arr
```
运行代码：
```bash
$ bash test21.sh #这样不用加上 sudo chmod o+x test21.sh 来 ./test21.sh 运行
7
```

**2. 大括号{ }**

1.文件名扩展
复制 t.txt 的内容到 t.back 中

```bash
$ vim test22.sh
```
输入代码：
```bash
#!/bin/bash

if [ ! -w 't.txt' ];
then
    touch t.txt
fi
echo 'test text' >> t.txt
cp t.{txt,back}
```
运行代码：
```bash
$ bash test22.sh
```
查看运行结果：
```bash
$ ls
$ cat t.txt
$ cat t.back
```
注意： 在大括号中，不允许有空白，除非这个空白被引用或转义。



代码块，

又被称为内部组，这个结构事实上创建了一个匿名函数（一个没有名字的函数）。<span style="background-color:#FFFF00">**然而，与“标准”函数不同的是，在其中声明的变量，对于脚本其他部分的代码来说还是可见的**</span>。

```bash
$ vim test23.sh
```
输入代码：
```bash
#!/bin/bash

a=123
{ a=321; }
echo "a = $a"
```
运行代码：
```
$ bash test23.sh
a = 321
```
<span style="color:red">与小括号不同 **变量 a 的值被更改了**</span>。


**3. 三、中括号[ ]**

1.条件测试

条件测试表达式放在```[ ]```中。下列练习中的```-lt``` (less than)表示小于号。

```bash
$ vim test24.sh
```
输入代码：
```bash
#!/bin/bash

a=5
if [ $a -lt 10 ]
then
    echo "a: $a"
else
    echo 'a>=10'
fi
```
运行代码：
```bash
$ bash test24.sh
a: 5
```
双中括号（```[[ ]]```）也用作条件测试（判断）

2.数组元素

在一个array结构的上下文中，中括号用来<span style="color:red">**引用数组中每个元素的编号**</span>。

```bash
$ vim test25.sh
```
输入代码：
```bash
#!/bin/bash

arr=(12 22 32)
arr[0]=10
echo ${arr[0]}
```
运行代码：
```bash
$ bash test25.sh
10
```


**4. 尖括号 < 和 >**

重定向

```test.sh > filename```：重定向test.sh的输出到文件 filename 中。如果 filename 存在的话，那么将会被覆盖。

```test.sh &> filename```：重定向 test.sh 的 stdout（标准输出）和 stderr（标准错误）到 filename 中。

```test.sh >&2```：重定向 test.sh 的 stdout 到 stderr 中。

```test.sh >> filename```：把 test.sh 的输出追加到文件 filename 中。如果filename 不存在的话，将会被创建。



**5. 竖线 |**

管道: 分析前边命令的输出，并将输出作为后边命令的输入。这是一种产生命令链的好方法。

```bash
$ vim test26.sh
```
输入代码：
```bash
#!/bin/bash

tr 'a-z' 'A-Z'
exit 0
```
现在让我们输送l```s -l```的输出到一个脚本中：
```bash
$ chmod 755 test26.sh
$ ls -l | ./test26.sh
```
输出的内容均变为了大写字母。

![](/img/post/shell/vertical_line.png)



**6.破折号-**

1.选项，前缀: 在所有的命令内如果想使用选项参数的话,前边都要加上“-”。

```bash
$ vim test27.sh
```

输入代码：
```bash
#!/bin/bash

a=5
b=5
if [ "$a" -eq "$b" ]
then
    echo "a is equal to b."
fi
```
运行代码：
```bash
$ bash test27.sh
a is equal to b.
```

2.用于重定向stdin或stdout

下面脚本用于备份最后24小时当前目录下所有修改的文件.
```bash
$ vim test28.sh
```
输入代码：
```bash
#!/bin/bash

BACKUPFILE=backup-$(date +%m-%d-%Y)
# 在备份文件中嵌入时间.
archive=${1:-$BACKUPFILE}
#  如果在命令行中没有指定备份文件的文件名,
#  那么将默认使用"backup-MM-DD-YYYY.tar.gz".

tar cvf - `find . -mtime -1 -type f -print` > $archive.tar
gzip $archive.tar
echo "Directory $PWD backed up in archive file \"$archive.tar.gz\"."

exit 0
```
运行代码：
```
$ bash test28.sh
$ ls
```

















**\c**: 再echo之后 让cursor继续在这一行，不重新开启一行 <br/>
**mv**: move and rename <br/>

让echo 和read 在一行
```bash

#Method 1 
echo -e "Enter the name of the file :\c" #\c is to keep the cursor on the line after echo, 不会重新开启一行
#-e flag: enable interpretation of backslash 如果不用-e, 会直接print \c 
read file_name 

#Method 2
read -p 'username: ' user_var #allow input on the same line, 如果不加flag -p 有error
echo "username $user_var"

```

```shell

cat /etc/shells #show which shell support your system (e.g. sh bash dash)

#sh: Bourne shell, #bash: Bourne against shell, is reinvented and improved after shell

touch hello.sh #生成hello
#.sh is not necessary, 没有.sh 也可以fine, has extension, editor will understand it is shell, give beautiful layout
#-rw- it doesn't include executable by default after generate by touch 

chmod +x hello.sh #grant execute access

./hello.sh #run shell script

```

Bash code (文件命名为hello.sh)
```bash
#! /bin/bash  #intepreter know it is bash script

#this is a comment
echo "Hello world" #print Hello world

```



## variable

变量赋值 都不加 ```$``` (即使前面出现也不加F), 引用加 ```$```

#### 变量定义

如果 ```variable``` 是一个变量的名字，那么 ```$variable``` 就是引用这个变量的值，即这变量所包含的数据。

```variable``` 事实上只是```{variable}``` 的简写形式。在某些上下文中 ```variable``` 可能会引起错误，这时候你就需要用```{variable}``` 了。


定义变量时，变量名不加美元符号（```$```，PHP语言中变量需要），如： ```myname="home"```

注意 <span style="background-color:#FFFF00">**变量名和等号之间不能有空格**</span>。同时，变量名的命名须遵循如下规则：

- 首个字符必须为字母（a-z，A-Z）。
- 中间不能有空格，可以使用下划线（_）。
- 不能使用标点符号。
- 不能使用bash里的关键字（可用help命令查看保留关键字）。
- 除了直接赋值，还可以用语句给变量赋值，如：```for file in `ls /etc````


#### 使用变量

变量名前加美元符号，如：

```bash
myname="home"
echo $myname 
echo ${myname}
echo ${myname}Good
echo $mynameGood #print nothing, because we dont have any mynameGood variable


myname="miao" #跟上面myname 是一个变量
echo ${myname}
```

output 

![](/img/post/shell/variable.png)


<span style="background-color:#FFFF00">**加花括号帮助解释器识别变量的边界**</span>，若不加，解释器会把mynameGood当成一个变量（值为空）, <span style="background-color:#FFFF00">**推荐给所有变量加花括号**</span>

已定义的变量可以重新被定义


#### 只读变量

使用 ```readonly``` 命令可以将变量定义为只读变量，<span style="color:red">**只读变量的值不能被改变**</span>。 下面的例子尝试更改只读变量，结果报错：

```bash
#!/bin/bash
myUrl="http://www.shiyanlou.com"
readonly myUrl
myUrl="http://www.shiyanlou.com"
```
运行脚本，结果如下：

```bash
/bin/sh: NAME: This variable is read only.
```


#### 特殊变量

**1.局部变量**

这种变量只有在代码块或者函数中才可见。

**2.环境变量**

这种变量将影响用户接口和 shell 的行为。

在通常情况下，每个进程都有自己的“环境”，这个环境是由一组变量组成的，这些变量中存有进程可能需要引用的信息。在这种情况下，shell 与一个一般的进程没什么区别。

**3.位置参数**

从命令行传递到脚本的参数：```0，1, 2，3...```

0就是脚本文件自身的名字，0就是脚本文件自身的名字，1 是第一个参数，2 是第二个参数，2是第二个参数，3 是第三个参数，然后是第四个。<span style="background-color:#FFFF00">**9 之后的位置参数就必须用大括号括起来了**</span>，比如，9之后的位置参数就必须用大括号括起来了，比如，```{10}，{11}，{12}```。

- ```$#``` ： 传递到脚本的**参数个数**
- ```$*``` ： 以一个单字符串显示所有向脚本传递的参数。与位置变量不同,此选项参数可超过 9个
- ```$$``` ： 脚本运行的当前进程 ID号
- ```$!``` ： 后台运行的最后一个进程的进程 ID号
- ```$@``` ： 与$*相同,但是使用时加引号,并在引号中返回每个参数
- ```$```： 显示shell使用的当前选项,与 set命令功能相同
- ```$?``` ： 显示最后命令的退出状态。 0表示没有错误,其他任何值表明有错误。

例子:

```bash
$ vim test30.sh
```
输入代码（中文皆为注释，不用输入）：
```bash
#!/bin/bash

# 作为用例, 调用这个脚本至少需要10个参数, 比如:
# bash test.sh 1 2 3 4 5 6 7 8 9 10
MINPARAMS=10

echo

echo "The name of this script is \"$0\"."

echo "The name of this script is \"`basename $0`\"."


echo

#-n STRING: true if the length of STRING is non-zero

if [ -n "$1" ]              # 测试变量被引用.
then
echo "Parameter #1 is $1"  # 需要引用才能够转义"#"
fi 

if [ -n "$2" ]
then
echo "Parameter #2 is $2"
fi 

if [ -n "${10}" ]  # 大于$9的参数必须用{}括起来.
then
echo "Parameter #10 is ${10}"
fi 

echo "-----------------------------------"
echo "All the command-line parameters are: "$*""

if [ $# -lt "$MINPARAMS" ]
then
 echo
 echo "This script needs at least $MINPARAMS command-line arguments!"
fi  

echo

exit 0
```
运行代码：

```bash
$ bash test30.sh 1 2 10

The name of this script is "test.sh".
The name of this script is "test.sh".

Parameter #1 is 1
Parameter #2 is 2
-----------------------------------
All the command-line parameters are: 1 2 10

This script needs at least 10 command-line arguments
```



**System Variable**: created and maintained by operating system, 通常是大写的
**User Variable**: created and maintained by user, 习惯是定义成小写的

```bash
#! /bin/bash
echo $BASH #give name of version 
echo $BASH_VERSION #give Bash version
echo $HOME #give home directory
echo current directory is $PWD  #give present working directory

#user defined, variable cannot start with number, 比如不能是定义变量为 10val
name=Mark
echo The name is $name #print The name is Mark

10val=10 #error
echo value $10val


```

## 运算

#### 算术运算符

<span style="color:red">原生bash不支持简单的数学运算</span>，但是可以通过其他命令来实现，例如 ```awk`` 和 ``expr``，``expr`` 最常用。

``expr`` 是一款表达式计算工具，使用它能完成表达式的求值操作.  <span style="color:red">**expr 只能用于整数计算**</span>，可以使用 ```bc``` 或者 ```awk``` 进行浮点数运算。


- 注意使用的反引号（esc键下边）
- 表达式和运算符之间<span style="background-color:#FFFF00">**要有空格**</span> ```$a + $b```写成```$a+$b```不行
- 条件表达式要放在方括号之间，并且<span style="background-color:#FFFF00">**要有空格**</span>```[ $a == $b ```]写成```[$a==$b]```不行
- 乘号（```*```）前边必须加反斜杠（```\```)才能实现乘法运算

```bash
#!/bin/bash

a=10
b=20

val=`expr $a + $b`
echo "a + b : $val"

val=`expr $a - $b`
echo "a - b : $val"

val=`expr $a \* $b`
echo "a * b : $val"

val=`expr $b / $a`
echo "b / a : $val"

val=`expr $b % $a`
echo "b % a : $val"

if [ $a == $b ]
then
   echo "a == b"
fi
if [ $a != $b ]
then
   echo "a != b"
fi
```
运行
```bash
$bash test.sh
a + b : 30
a - b : -10
a * b : 200
b / a : 2
b % a : 0
a != b
```

浮点数运算: 先安装bc

```shell
sudo apt-get update
$ sudo apt-get install bc
```

```bash
#!/bin/bash

raduis=2.4

pi=3.14159

girth=$(echo "scale=4; 3.14 * 2 * $raduis" | bc)

area=$(echo "scale=4; 3.14 * $raduis * $raduis" | bc)

echo "girth=$girth"

echo "area=$area"
```




#### 关系运算符

| 符号 | 解释 |
| :---: | :--- | 
| ```-eq``` | is equal to ```if[ $a -eq $b ]``` |
| ```-ne``` | is not equal to ```if[ $a -ne $b]``` |
| ```-gt``` | is greater than ```if[ $a -gt $b ]```|
| ```>``` | is greater than ```(($a > $b))``` |
| ```-ge``` | is greater than or equal to ```if[ $a -ge $b ]```|
| ```>=``` | is greater than or equal to ```(($a >= $b))```  or ``` [[ $a>=$b ]]``` (如果没有两个```[[]]```, always true )|
| ```-lt``` | is less than  ```if[ $a -lt $b ]```|
| ```<``` | is less than ```(($a < $b)``` |
| ```-le``` | is less than to  ```if[ $a -le $b ]``` |
| ```<=``` | is less than or equal to ```(($a <= $b))``` |


注意： 多于numeric, ```>=, ==, >, >=``` 比较需要用 双小括号，```(($count > 10))```, 多于string ```==, !=, >, <```, 需要用双中括号 ```[[]]```


<span style="background-color:#FFFF00">**关系运算符只支持数字，不支持字符串，除非字符串的值是数字**</span>。

```bash
vim test2.sh
```

```bash
#!/bin/bash

a=10
b=20

if [ $a -eq $b ]
then
   echo "$a -eq $b : a == b"
else
   echo "$a -eq $b: a != b"
fi
````
运行

```bash
$bash test2.sh
10 -eq 20: a != b
```

#### 逻辑运算符


```bash
#!/bin/bash
a=10
b=20

if [[ $a -lt 100 && $b -gt 100 ]]
then
   echo "return true"
else
   echo "return false"
fi

if [[ $a -lt 100 || $b -gt 100 ]]
then
   echo "return true"
else
   echo "return false"
fi
```

结果
```bash
return false
return true
```



#### Logical And



```shell
#! /bin/bash

age=25

#Method 1 && 
if [ $age -gt 18 ] && [ $age -lt 30 ] # 18 < age < 30 
then 
    echo "valid age"
else 
    echo "age not valid"
fi 



if [[ $age -gt 18 && $age -lt 30 ]] # require double [[]]
then 
    echo "valid age"
else 
    echo "age not valid"
fi 


#Method 2 -a: stands for and 

age=25

if [ $age -gt 18 -a $age -lt 30 ] # 18 < age < 30 
then 
    echo "valid age"
else 
    echo "age not valid"
fi 



```



#### Logical Or



```bash
#! /bin/bash


age=60

#Method 1 || 
if [ $age -gt 18 ] || [ $age -lt 30 ] # 18 < age < 30 
then 
    echo "valid age"
else 
    echo "age not valid"
fi 

if [[ $age -eq 18 || $age -eq 30 ]] # require double [[]]
then 
    echo "valid age"
else 
    echo "age not valid"
fi 


#Method 2 -o: stands for or 

age=60

if [ $age -gt 18 -o $age -lt 30 ] # 18 < age < 30 
then 
    echo "valid age"
else 
    echo "age not valid"
fi 



```



#### 字符串运算符


| 符号 | 解释 |
| :---: | :--- | 
| ```=``` | is equal to ```if[ $a = $b ]``` 跟```==``` 一样的，<span style="background-color:#FFFF00">一定留空格在 ```$a``` , ```$b``` 和 ```[]``` 之间</span> |
| ```==``` | is equal to ```if[ $a == $b ]```|
| ```!=``` | is not equal to ```if[ $a != $b ]```|
| ```<``` | is less than, in ASCII alphabetical order ```if [[ $a < $b ]]```, <span style="background-color:#FFFF00">**注意对比string 大小用 两个bracket**</span>|
|```>``` | is greater than, in ASCII alphabetical order ```if[[ $a > $b ]]```|
|```-z```|  if string is null, return true if zero length |
| ```-n``` | if string is null, return true if not zero length  |
| ```str``` | 检测字符串是否为空, 不为空返回true | 

```bash
a="abc"
b="efg"

if [ $a = $b ]
then
   echo "$a = $b : a == b"
else
   echo "$a = $b: a != b"
fi
if [ -n $a ]
then
   echo "-n $a : The string length is not 0"
else
   echo "-n $a : The string length is  0"
fi
if [ $a ]
then
   echo "$a : The string is not empty"
else
   echo "$a : The string is empty"
fi
```
结果
```bash
abc = efg: a != b
-n abc : The string length is not 0
abc : The string is not empty
```

#### 文件测试运算符

![](/img/post/shell/file.png)


```bash
#!/bin/bash

file="/home/shiyanlou/test.sh"
if [ -r $file ]
then
   echo "The file is readable"
else
   echo "The file is not readable"
fi
if [ -e $file ]
then
   echo "File exists"
else
   echo "File not exists"
fi
```
结果
```bash
The file is readable
File exists
```

## 流程控制



#### if

[if primary expressions](http://tldp.org/LDP/Bash-Beginners-Guide/html/sect_07_01.html#sect_07_01_01)


**if 语句语法格式**：

```
if condition
then
    command1 
    command2
    ...
    commandN 
fi
```

**2. if else 语法格式**：

```
if condition
then
    command1 
    command2
    ...
    commandN
else
    command
fi
```


**3. if-elif-else 语法格式**：

```
if condition1
then
    command1
elif condition2 
then 
    command2
else
    commandN
fi
```

<span style="background-color:#FFFF00">**if else语句经常与```test```命令结合使用**</span>

```bash
num1=$[2*3]
num2=$[1+5]
if test $[num1] -eq $[num2]
then #
    echo 'Two numbers are equal!'
else #
    echo 'The two numbers are not equal!'
fi
```
输出结果：
```
Two numbers are equal!
```


以下实例判断两个变量是否相等：

```bash
a=10
b=20
if [ $a == $b ]
then #
   echo "a == b"
elif [ $a -gt $b ]
then #
   echo "a > b"
elif [ $a -lt $b ]
then #
   echo "a < b"
else #
   echo "Ineligible"
fi
```
输出结果：

```bash
a < b
```


<span style="background-color:#FFFF00">**注意if 跟方括号```[```之间要有空格**</span>

```bash
#! /bin/bash

#syntax
if [ condition ] 
then  #
    statement
fi #end of if statement 

count = 10 
if [$count -eq  10 ]
then #
    echo "condition is true"
fi 


#or 
if (($count >  9))
then #
    echo "condition is true"
else #
    echo "condition is false"
fi 

#or
if (($count ==  10)) #注意不能写成 [ $count == 10 ]会报错
then #
    echo "condition is true"
else #
    echo "condition is false"
fi 




#string, == 和 = 都是一样的，是不是相等
word=abc
#string 写成 word="abc" or word=abc 都可以
if [ $word == "abc"]
then #
    echo "condition is true"
fi

word="a"
if [[ $word == "b" ]]
then #
    echo "condition b is true"
elif [[ $word == "a" ]]
then #
    echo "condition a is true"
else #
    echo "condition is false"
fi

```

<span style="background-color: #FFFF00">**注意**</span>： 多于numeric, >=,  ==, >, >= 比较需要用 双小括号，(($count > 10)), 多于string ==, !=, >, <, 需要用双中括号 [[]]

<span style="background-color: #FFFF00">**注意**</span>： 写if condition 需要让括号和里面内容有空格，比如 if [[ $word == "a" ]] 是可以的，如果是 if [[ $word == "a"]] 是错误的 

![](/img/post/shell/if.png)



#### For Loop

for循环一般格式为：
```
for var in item1 item2 ... itemN
do
    command1
    command2
    ...
    commandN
done
```
例如，顺序输出当前列表中的数字：
```
for loop in 1 2 3 4 5
do
    echo "The value is: $loop"
done
```
输出结果：

```
The value is: 1
The value is: 2
The value is: 3
The value is: 4
The value is: 5
```
顺序输出**字符串**中的字符：
```
for str in This is a string
do
    echo $str
done
```
输出结果：
```
This
is
a
string
```

```bash
#! /bin/bash

for i in 1 2 3 4 5 #number separated by spaces 
do #
    echo $i
done 

#

for i in {1..10} #number from 1 to 10
do  #
    echo $i
done 

#can only use for bash over 4.0
for i in {1..10..2} #number from 1 to 10, increase by 2
do  #
    echo $i
done 

#

for (( i=0; i<10; i++ ))
do #
    echo $i
done 

#
for (( i=0; i<10; i+=2 ))
do #
    echo $i
done 


```


![](/img/post/shell/forloop.png)

![](/img/post/shell/forloop1.png)



#### For loop to execute command



```bash
#! /bin/bash

for command in ls pwd date #execute command one by one
do  #
    echo "----------$command-----------"
    $command 
done 

#print all the directory inside current folder
for item in * #* every item in the directory
do #
    if [ -d $item ]  #if it is directory,  if [ -f $item ] if it is file
    then #
        echo $item 
    fi 
done 


```

## While


```bash
#! /bin/bash

#sytax 
while [condition] 
do  #
    command1
    comand2
    comand3
done #


n=1 
while [ $n -le 10 ] #or (( $n <= 10 ))
do  #
    echo $n 
    n=$(( $n + 1 )) # n= 不能有空格
    # or  let n++ or (( n++ ))
done  #


```

```bash
#!/bin/bash
int=1
while(( $int<=5 ))
do #
    echo $int
    let "int++"
done #
```
运行脚本，输出：
```
1
2
3
4
5
```
如果int小于等于5，那么条件返回真。int从1开始，每次循环处理时，int加1。运行上述脚本，返回数字1到5，然后终止。
使用了 Bash let 命令，它用于执行一个或多个表达式，变量计算中不需要加上 $ 来表示变量

while循环可用于读取键盘信息。下面的例子中，输入信息被设置为变量MAN，按<Ctrl-D>结束循环。

```bash
echo 'press <CTRL-D> exit'
echo -n 'Who do you think is the most handsome: '
while read MAN
do #
    echo "Yes！$MAN is really handsome"
done #
```


#### 无限循环

无限循环语法格式：
```bash
while :
do #
    command
done #

# 或者
while true
do #
    command
done #

# 或者

for (( ; ; ))

```




## Until Loop

until循环执行一系列命令直<span style="background-color:#FFFF00">**至条件为真时停止**</span>。 until循环与while循环在处理方式上刚好相反。 一般while循环优于until循环，但在某些时候—也只是极少数情况下，until循环更加有用。 until 语法格式:
```bash
until condition
do #
    command
done #
```

条件可为任意测试条件，测试发生在循环末尾，因此循环至少执行一次—请注意这一点。

```bash
#! /bin/bash

#syntax
until [condition]
do  #
    command1
    command2
    command3
done #

n=1
until [ $n -ge 10 ] #or use (( $n > 10 ))
do 
    echo $n
    n=$(( n + 1 )) #or use (( n++ ))
done 


```


## Case


Shell case语句为多选择语句。可以用case语句匹配一个值与一个模式，如果匹配成功，执行相匹配的命令。case语句格式如下：
```bash
case 值 in
模式1) #
    command1
    command2
    ... #
    commandN
    ;;
模式2） #
    command1
    command2
    ... #
    commandN
    ;; #
esac
```


- 取值后面必须为单词<span style="background-color:#FFFF00">**in**</span>，每一模式必须以<span style="background-color:#FFFF00">**右括号结束**</span>。取值可以为<span style="background-color:#FFFF00">**变量或常数**</span>。匹配发现取值符合某一模式后，其间所有命令开始执行直至 <span style="background-color:#FFFF00">**;;**</span>。
- 取值将检测匹配的每一个模式。一旦模式匹配，<span style="color:red">则执行完匹配模式相应命令后不再继续其他模式</span>。如果无一匹配模式，使用星号 ```*``` 捕获该值，再执行后面的命令。
- 它需要一个```esac```（就是case反过来）作为结束标记，每个case分支<span style="background-color:#FFFF00">**用右圆括号**</span>，用<span style="background-color:#FFFF00">**两个分号表示break**</span>。



下面的脚本提示输入1到4，与每一种模式进行匹配：
```bash
echo 'Enter a number between 1 and 4:'
echo 'The number you entered is:'
read aNum
case $aNum in #
    1)  echo 'You have chosen 1'
    ;;  #
    1)  echo 'You have chosen 2'
    ;; #
    1)  echo 'You have chosen 3'
    ;; #
    1)  echo 'You have chosen 4'
    ;; #
    *)  echo 'You did not enter a number between 1 and 4'
    ;;
esac
```
输入不同的内容，会有不同的结果，例如：

```
Enter a number between 1 and 4:
The number you entered is:
3
You have chosen 3
```

不光可以判断strict letters/character, 可以判断是不是符合一些pattern

```bash
#! /bin/bash

vehicle=$1

case $vehicle in
    "car" )
        echo "Rent of $vehicle is 100 dolar" ;;
     "van" )
        echo "Rent of $vehicle is 80 dolar" ;;
     "bicycle" )
        echo "Rent of $vehicle is 5 dolar" ;;
     "truck" )
        echo "Rent of $vehicle is 150 dolar" ;;
    * ) #default case  
        echo "Unknown vechicle" ;;
esac


echo -e "Enter some character : \c"
read value 

#pattern
case $value in
    [a-z] ) #enter smaller character 
        echo "User entered $value a to z" ;;
    [A-Z] )
        echo "User entered $value A to Z" ;;
    [0-9] ) #enter integer
        echo "User entered $value 0 to 9" ;;
    ? ) #? means every special character which is one letter character
        echo "User entered $value special character" ;;
    * ) #more than one character， 只有一位的character 被? capture了 
        echo "Unknown input" ;;
esac

#如果运行上面的，发现输入K 跑到case a - z 里面了
# 需要run command LANG=C 会fix 让 K 跑到 case A-Z
#'LANG' enviroment variable indicates the language/locale and encoding where "C" is the language setting


```




## Break / Continue



break命令允许跳出所有循环（终止执行后面的所有循环）。 下面的例子中，脚本进入死循环直至用户输入数字大于5。要跳出这个循环，返回到shell提示符下，需要使用break命令。

```bash
#!/bin/bash
while :
do #
    echo -n "Enter a number between 1 and 5:"
    read aNum
    case $aNum in #
        1|2|3|4|5) echo "The number you entered is $aNum!"
        ;; #
        *) echo "The number you entered is not between 1 and 5! game over!"
            break
        ;; #
    esac #
done
```
执行以上代码，输出结果为：

```
Enter a number between 1 and 5:3
The number you entered is 3!
Enter a number between 1 and 5:7
The number you entered is not between 1 and 5! game over!
```

continue命令与break命令类似，只有一点差别，它不会跳出所有循环，仅仅跳出当前循环。 对上面的例子进行修改：

```bash
#!/bin/bash
while :
do #
    echo -n "Enter a number between 1 and 5: "
    read aNum
    case $aNum in #
        1|2|3|4|5) echo "The number you entered is $aNum!"
        ;; #
        *) echo "The number you entered is not between 1 and 5!"
            continue
            echo "game over"
        ;; #
    esac #
done
```
运行代码发现，当输入大于5的数字时，该例中的循环不会结束，语句 echo "Game is over!" 永远不会被执行。




```bash
#! /bin/bash

for (( i=1 ; i<=10 ; i++ ))
do  #
    if [ $i -gt 5 ]
    then  #
        break #break the loop 当 i > 5 
    fi #
    echo "$i"
done

#

for (( i=1 ; i<=10 ; i++ )) 
do  #
    if [ $i -eq 3 -o $i -eq 6 ]
    then  #
        continue #continue if i = 3 or i = 6,  不print
    fi #
    echo "$i"
done


```
















## Read User Input  


```bash

#! /bin/bash

echo "Enter name: "
read name #input is recorded into variable name, input 是在Next line
echo "Entered name: $name"

echo "Enter names: "
read name1 name2 name3 #separate variables by space, input 在next line
echo "Names: $name1, $name2, $name3"


read -p 'username: ' user_var #allow input on the same line, 如果不加flag -p 有error
echo "username $user_var"

#want to make input silence, not show what is typing 

read -sp "password: " pass_var
echo  #避免print on the same line after slience input
echo "password: " $pass_var

#read in array -a
echo "Enter names "
read -a names 
echo "Names:  ${names[0]} ${names[1]}" #需要curly bracket

#enter read without any variable, input go to builtin variable $REPLY
echo "Enter names"
read 
echo "Names: $REPLY"

```



## Pass Arguments to Bash Script



```bash

#! /bin/bash

echo $0 $1 $2 $3 ' > echo $1 $2 $3 ' #$0 is script name, first argument pass $1, second argument pass to $2

#./hello.sh Mark Tom John
# print Mark Tom John > echo  $1, $2 $3

#pass variable into array, declare variable args, $@ is used to store argument as array
args=("$@") #比一定非要用args, 可以用ch=("$@") 也是可以的
echo ${args[0]}, ${args[1]} 
#./hello.sh Mark Tom John
# print Mark Tom, args[0] 是 argument 不是filename 

echo $@ #print all the argument will be printed. 有几个pass, 有几个print
echo $# #print number of argument passed

```

<span style="background-color: #FFFF00">Difference between pass into array and variable</span>,$0 是 file name, 但是array index 0 是first argument





## File Test Operator

if statement flag: <br/>
**-e**: check if exist <br/>
**-f**: check if it is file <br/>
**-d**: check if it is directory  <br/>
**-b**: check if it is blocked special file(txt) <br/>
**-c**: check if it is character special file (pic, video) <br/>
**-s**: check if file is empty <br/>
**-r**: check if file has read permission <br/>
**-w**: check if file has write permission <br/>
**-x**: check if file has execute permission <br/>




```bash
#! /bin/bash

echo -e "Enter the name of the file :\c" #\c is to keep the cursor on the line after echo, 不会重新开启一行
#-e flag: enable interpretation of backslash 如果不用-e, 会直接print \c 
read file_name 

if [ -e $file_name ] #-e check if file exist
then  
    echo "$file_name found"
else
    echo "$file_name not found"
fi


if [ -f $file_name ] #-f check if it is normal file when file exisit
then  
    echo "$file_name is file"
else
    echo "$file_name not file"
fi


if [ -d $dir_name ] #-d check if it is normal file when file exisit
then  
    echo "$dir_name is file"
else
    echo "$dir_name not file"
fi


#block special file: contain some text or data like txt  
# character special file: binary file e.g. picture video 

if [ -b $file_name ] #-b check if it is blocked special file
then  
    echo "$file_name is blocked special file"
fi 


if [ -c $file_name ] #-c check if it is character special file
then  
    echo "$file_name is character special file"
fi 

if [ -s $file_name ] #-s check if file is empty or not 
then  
    echo "$file_name is character special file"
fi 

```


## Append Output to text file

cat >> $file_name: append to the file
cat > $file_name: overwrite the file


```bash
#! /bin/bash

echo -e "Enter the name of the file :\c" #\c is to keep the cursor on the line after echo, 不会重新开启一行
#-e flag: enable interpretation of backslash 如果不用-e, 会直接print \c 
read file_name 

if [ -f $file_name ] #-f check if it is normal file when file exisit
then  
   if [ -w $file_name ] #-w check if file has write permission 
   then 
        echo "Type some text data. To quit press ctrl + d"
        cat >> $file_name
   else 
        echo "The file not have write permission"
   fi
else
    echo "$file_name not file"
fi


#如果file 没有write permission, run command
chmod +w $filename 

```






## Arithmetic operations

**let** is a builtin function of Bash that allows us to do simple arithmetic. 

```bash
#!/bin/bash
# Basic arithmetic using let

let a=5+4  #不能有space
echo $a # 9

let "a = 5 + 4" #allow us to space out the expression 
echo $a # 9

let a++
echo $a # 10

let "a = 4 * 5"
echo $a # 20

let "a = $1 + 30"  #include the parameter from argument 
echo $a # 30 + first command line argument


```


**expr** is similar to let except instead of saving the result to a variable it instead  <span style="color: red">prints the answer</span>. Unlike let you don't need to 不用引号. You also must 必须有space between items of the expression. It is also common to use expr within command substitution to save the output to a variable.

```bash 
#! /bin/bash


# ruun ./hello.sh 12

expr 5 + 4 #print 9  必须有space 并且没有quote

expr "5 + 4" #print 5 + 4
expr 5+4 #print 5+4, 没有space 不会evaluated 直接print
expr 5 \* $1  #60 (因为5乘以12)， $1 first argument
expr 11 % 2  #1
a=$( expr 10 - 3 ) #7, using expr within command substitution in order to save the result to the variable a.
echo $a # 7
```

![](/img/post/shell/expr.png)

**Double Parenthese**  We do so by using double brackets like so: $(( expression ))

```bash 
#!/bin/bash
# Basic arithmetic using double parentheses
a=$(( 4 + 5 )) #without the need for quotes.
echo $a # 9

a=$((3+5)) #It works just the same if we take spacing out.
echo $a # 8

b=$(( a + 3 )) #We may include variables without the preceding $ sign.
echo $b # 11

b=$(( $a + 4 )) # Variables can be included with the $ sign if you prefer.
echo $b # 12

(( b++ )) #let b++ 一样的功能
echo $b # 13

(( b += 3 ))
echo $b # 16

a=$(( 4 * 5 )) #we don't need to escape like expr 
echo $a # 20

```




Example: 
```bash
#! /bin/bash


echo 1+1 #echo take anything after echo command as string by default, print 1+1 not 2

num1=20 
num2=5

echo  $(( num1 + num2 )) #需要两个小括号 加上 $
echo  $(( $num1 + $num2 )) #与上面一样的 

echo  $(( $num1 - $num2 )) 
echo  $(( $num1 * $num2 )) 
echo  $(( $num1 / $num2 )) 
echo  $(( $num1 % $num2 )) 

echo  $( expr $num1 - $num2 ) 
echo  $( expr $num1 \* $num2 ) #当expr 需要 \* escape character
echo  $( expr $num1 / $num2 )
echo  $( expr $num1 % $num2 )

 

```






## Length of a varible



```bash
#! /bin/bash

#syntax: ${#variable}


a='Hello World'
echo ${#a} # 11

b=4953
echo ${#b} # 4

```



## Floating Point Math Operations

bc: basic calulator, 除法时候需要定义scale, 否则给的结果不准备，bc 还可以用square root 之类的 <br/>
bc -l: calling the math library

```bash
#! /bin/bash

num1=20.5
num2=5

echo $(( num1 + num2 )) #会有error: non integer argument, bash doesn't support floating point,

#用bc basic calculator

echo "20.5+5" | bc #whatever the command you use in the left hand side will be treated as input for right hand side command
echo "20.5-5" | bc #15.5
echo "20.5*5" | bc  #102.5
echo "20.5/5" | bc #4  not proper value of division
echo "scale=2; 20.5/5" | bc #4.10 scale how many digits / preicision of result
echo "20.5%5" | bc #.5 

echo "$num1 + $num2" | bc 
echo "$num1 - $num2" | bc 

num=27
#calculate square root of 27 and keep it in 2 digits
echo "scale=2; sqrt($num)" | bc -l #-l, 表示calling the math library

echo "scale=2; 3^3" | bc -l #计算的3的立方

```



## Array 

```@```: all of the array <br/>
```!```: index of the array <br/>
```#```: length of the array <br/>

Array: 必须现在array size 是3, 可以在index =6 的位置initialized, 打印array时候，没有被initialized的会被ignored, gap is okay

```bash
#! /bin/bash

os=('ubuntu' 'windows' 'kali')
echo "${os[@]}" # [@] print all element of the array

echo "${os[0]} , ${os[1]}" # print 第一个，第二个element

echo "${!os[@]}" #! print 0  1 2 , print the index of the array

echo "${#os[@]}" # print 3,  #print the length of the array

os[3] = 'mac' #append to the array, array 变成'ubuntu' 'windows' 'kali' 'mac'
echo "${#os[@]}" # print 4,

#remove from array
unset os[2] #remove 第三个element, array 变成'ubuntu' 'windows' 'mac'
echo "${!os[@]}" #! print 0  1 3 , print the index of the array
echo "${#os[@]}" # print 3,  #print the length of the array


#比如现在array 长度是3, 我们可以append 到第6的位置, array 一些位置是uninstalized的
os=('ubuntu' 'windows' 'kali')
os[6]='iam'

echo "${os[@]}" # [@] print all element of the array
echo "${#os[@]}" #print 4
echo "${!os[@]}" #print 0 1 2 6

echo "${os[3]} , ${os[6]}" # print , iam

string=abcdefghi
echo "${string[@]}" #take string as array, 
echo "${string[0]}" #print abcdefghi, 
echo "${string[1]}" #print nothing
echo "${#string[@]}" #print 1
#why this happening? because you treat variable as array, the value of the variable assign at the index 0

```





## Sleep, Open Terminal



```bash
#! /bin/bash

n=1 
while [ $n -le 10 ] #or (( $n <= 10 ))
do 
    echo $n 
    n=$(( $n + 1 )) # n= 不能有空格
    # or  let n++ or (( n++ ))
    sleep 1 #sleep 1 seconds then execute 
done 

#open 3 geno-terminals and 3 terminals
n=1 
while [ $n -le 3 ] #or (( $n <= 10 ))
do 
    echo $n 
    n=$(( $n + 1 )) # n= 不能有空格
    # or  let n++ or (( n++ ))
    geno-terminal & #& means end symbol
    xterm & 
done 


```



## Read File from while loop



```bash
#! /bin/bash

while read p 
do 
    echo $p
done < hello.sh #< input read direction, read hello.sh line by line


#Method 2
cat hello.sh | while read p #whatever output from cat as the input for the while command
do 
    echo $p
done 

#有时候file 可能有special character/indentation 上面两种方法都读不了，用ifs
#ifs: internal field separator, used by Shell to determine how to do the word splitting

while IFS= read -r line # IFS= 和read 之间有空格，means we assign space to IFS not read
#-r flag means prevent backslash escape from being interpreted 
do 
    echo $line
done < hello.sh 

while IFS=' ' read -r line #也可以用IFS=' '

```








## Select loop

Select: generate easy menu when you write script require menu then you can use select loop. Often used in cases 

```bash
#! /bin/bash

#syntax
select varName in list 
do  
    command1
    command2
    command3
done

select varName in mark john tom ben 
do  
    echo " $varName selected"
done

#print and prompt ask you to enter a number from the group
# 1） mark 
# 2) john
# 3) tom
# 4) ben
#输入3， print tom selected 


select name in mark john tom ben 
do  
    case $name in 
        mark) 
            echo "mark selected ";;
        john )
            echo "john selected"
        tom) 
            echo "tom selected ";;
        ben )
            echo "ben selected"
        *)
            echo "Error please the no. between 1..4"
        esac
done



```




## Function

By default, every variable defined is global variable. It means it can accessed in anywhere in script. Local can be used inside function only 

**$?**: 表示previous function return value <br/>
**$1 $2**: 在function 内部表示传入function 的参数
**local**: 定义在function 的variable为局部变量，只能在function 内用

```bash
#! /bin/bash

#syntax 1
function name(){
    command
}

#syntax 1
name (){
    command 
}

function Hello(){
    echo "Hello"
}

quit(){
    exit
}

quit #如果quit 在 Hello 前面，会quit 的script instead of print Hello
Hello #run the function


#Pass Argument into Function

function print(){
    local name=$1 #$1 first argument, 这个name is different from the global variable name (Tom)
    echo "$name" 
}

name="Tom"

echo "The name is $name : Before" #print Tom

print Max #Hello passed into function 

echo "The name is $name : After" #print Tom, 如果不在function 里面加上local, 会print Max, 因为function 里面改了name值






#example

usage(){
    echo "You need to provide an argument :"
    echo "usage : $0 file_name"
}

is_file_exist(){
    local file=$1 #$1 first argument provided by function
    if [[ -f $file ]] 
        then 
            echo "found file"    
    fi
    [[ -f $file ]] && return 1  || return 0 #如果file 存在return 1, 否则return 0 
    
}

[[ $# -eq 0 ]] && usage #if 没有 passing argument print usage message

is_file_exist $1 #$1 first argument provided by script
if [ $? -ne 0 ]  # $? 表示previous function return value
then
    echo "File found"
else 
    echo "File not found"
fi


#可以override function, allow to create wrapper

ls () {
command ls -lh #必须加command keyword 
}
#if we didn't put the keyword command in front of ls, would end up in an endless loop. Even though we are inside the function ls when we call ls it would have called another instance of the function ls which in turn would have done the same and so on.

ls

```



## Readonly Command

Read-only Command can be used with variables and functions, it can make variabls or functions read-only so they cannot be overwritten

```bash
#! /bin/bash

var=31
readonly var 

var=50 #give warning, var:  readonly variable

echo "var => $var" #print var => 31 即使我们尝试改，但是改不了

hello(){
    echo "Hello World"
}

readonly -f hello #需要 -f flag 定义read only function 


hello(){ #会有warning: hello: readonly function
    echo "Hello World Again"
}

readonly #如果只写readonly, it will print the list of all readonly builtin variables
readonly -p #与readonly 给的结果一样
readonly -f #see all the readonly functions


```
把hello function readonly, 再call readonly -f, 会显示出hello function 

![](/img/post/shell/readonly.png)


## Signals, Traps

$$ print the PID of the script itself

ctrl c: interrupt signal: sig-in, press ctrl+c to terminate <br/>
ctrl z: suspend signal, sig ts TP: script is running and by press ctrl + z stop running <br/>
kill -9 PID: 知道PID的话，也可以直接kill process

some unexpected behavior or signal can come to interrupt the execution of the script. **Trap command** provides the script to capture and interrupt and then clean it up within tte script. 可以捕获interrupt signal do some thing before exiting out

Except for Trap: cannot catch sigkill and sigstop command

```bash
#! /bin/bash

trap "echo Exit signal is detected" SIGINT #SIGINT 是 2, 可以用2代替SIGINT, 
#上面表示whenever recieve signal 2(Interrupt), it need to execute this command to echo

trap "echo Exit signal is detected" SIGKILL 
#上面表示whenever recieve signal 9(SIGKILL), 不会echo 上面的 因为SIGKILL 不能被trap catch



echo "pid is $$"
while (( COUNT < 10 ))
do 
    sleep 10
    (( COUNT ++ ))
    echo $COUNT
done 
exit 0



man 7 signal #show signal and vaue


```

![](/img/post/shell/signalstraps.png)


```bash


#! /bin/bash


#e.g. 1 
trap "echo Exit command is detected" 0 #signal 0 means success
#上面表示whenever recieve signal 0, it need to execute this command to echo 

echo "Hello world"

exit 0 #0 is success signal
#print
#"Hello world" "
# "Exit command is detected"

#e.g. 2 when recieve signal delete file

file="home/test/Desktop/file.txt"
trap "rm -f $file && echo file deleted;exit" 0 2 15 #combine 两个 command remove file and exit, 当遇到SIGNAL 0(sucess), 2(SIGINT), 15(terminate)


```

当一个terminal 运行file, 另一个terminal call trap, 会print 额外的信息，remove 额外的信息，call trap - signal_listed_in_file


![](/img/post/shell/signalstraps2.png)

![](/img/post/shell/signalstraps3.png)



## Debug

Method 1: add -x flag  when running shell

```bash
#! /bin/bash

echo "pid is $$"
while (( COUNT < 10 ) #少一个括号， run ./hello.sh error 显示在done 那行
do 
    sleep 10
    (( COUNT ++ ))
    echo $COUNT
done 

#run script 
bash -x ./hellp.sh  
#print everything for the code when code runing
#if something is wrong, you will be strictly aware of your script is not working in which line

```



Method 2: add x at the end of bash path

```bash
#! /bin/bash -x 
#print everything for the code when code runing
#if something is wrong, you will be strictly aware of your script is not working in which line


echo "pid is $$"
while (( COUNT < 10 ) #少一个括号， run ./hello.sh error 显示在done 那行
do 
    sleep 10
    (( COUNT ++ ))
    echo $COUNT
done 

#run script 
./hellp.sh  

```


Method 3: Set x in the file

```bash
#! /bin/bash 
set -x #activate debugging from writting set -x

echo "pid is $$"
set +x #deactivate debugging from writting set +x


while (( COUNT < 10 ) #少一个括号， run ./hello.sh error 显示在done 那行
do 
    sleep 10
    (( COUNT ++ ))
    echo $COUNT
done 

#run script 
./hellp.sh  

```







