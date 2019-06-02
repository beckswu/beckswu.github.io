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





## Logical And



```bash
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



## Logical Or



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



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```


## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```



## Append Output to text file



```bash
#! /bin/bash


```




## Append Output to text file



```bash
#! /bin/bash


```




## Append Output to text file



```bash
#! /bin/bash


```



![](/img/post/linux/watch.png)







