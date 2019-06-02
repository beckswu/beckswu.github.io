---
layout:     post
title:      "Shell Script - 笔记"
subtitle:   "Shell Script learning note "
date:       2019-06-01 22:00:00
author:     "Becks"
header-img: "img/post-bg2.jpg"
catalog:    true
tags:
    - Shell Script
    - 学习笔记
---


**\c**: 再echo之后 让cursor继续在这一行，不重新开启一行 <br/>
**mv**: move and rename <br/>


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



## if



```bash
#! /bin/bash

#syntax
if[ condition ] 
then 
    statement
fi #end of if statement 

count = 10 
if [$count -eq  10 ]
then 
    echo "condition is true"
fi 


#or 
if (($count >  9))
then 
    echo "condition is true"
else 
    echo "condition is false"
fi 

#or
if (($count ==  10)) #注意不能写成 [$count == 10 ]会报错
then 
    echo "condition is true"
else 
    echo "condition is false"
fi 




#string, == 和 = 都是一样的，是不是相等
word=abc
#string 写成 word="abc" or word=abc 都可以
if [ $word == "abc"]
then 
    echo "condition is true"
fi

word="a"
if [[ $word == "b" ]]
then 
    echo "condition b is true"
elif [[ $word == "a" ]]
then
    echo "condition a is true"
else 
    echo "condition is false"
fi

```

<span style="background-color: #FFFF00">**注意**</span>： 多于numeric, >=,  ==, >, >= 比较需要用 双小括号，(($count > 10)), 多于string ==, !=, >, <, 需要用双中括号 [[]]

<span style="background-color: #FFFF00">**注意**</span>： 写if condition 需要让括号和里面内容有空格，比如 if [[ $word == "a" ]] 是可以的，如果是 if [[ $word == "a"]] 是错误的 

![](/img/post/shell/if.png)



## File Test Operator


```bash
#! /bin/bash

echo "Enter the name of the file :\c" #\c is to keep the cursor on the line after echo, 不会重新开启一行

```


## cp

CP: copy and paste

```shell

cp options source destination

cp file1.txt file2.txt #if file2.txt not exist will creat file2.txt. Content from file1 will copy to file2 

cp file1.txt dir1 # 把file1.txt copy 到directory 1

cp file1.txt file2.txt dir1 # 把file1.txt 和file2.txt copy 到directory 1

cp -i  file1.txt file2.txt dir1 #如果directory 1 里面有file1.txt -i 会ask 是否要overwrite, 选n, 就会只copy file2 不会copy file1

cp ../f1.txt ../f2.txt . #从parent directory copy f1.txt和f2.txt 到现在directory, 因为没有-i, 会overwrite

cp dir1 dir3 # error， 因为dir1 有文件，不能被copy
cp -R dir1 dir3 # -R means recursive copy,copy everything from dir1 to dir3
#whenever destination (dir3) doesn't exist, it create destination and copy all content from source
#如果存在destination, 只copy paste
cp -vR dir1 dir3 #显示详细的copy 哪些文件


```


![](/img/post/linux/watch.png)







