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


## Case

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



## Array 

@: all of the array <br/>
!: index of the array <br/>
#: length of the array <br/>

Array: 必须现在array size 是3, 可以在index =6 的位置initialized, 打印array时候，没有被initialized的会被ignored, gap is okay

```bash
#! /bin/bash

os=('ubuntu' 'windows' 'kali')
echo "${os[@]}" # [@] print all element of the array

echo "${os[0]} , ${os[1]}" # print 第一个，第二个element

echo "{!os[@]}" #! print 0  1 2 , print the index of the array

echo "${#os[@]}" # print 3,  #print the length of the array

os[3] = 'mac' #append to the array, array 变成'ubuntu' 'windows' 'kali' 'mac'

#remove from array
unset os[2] #remove 第三个element, array 变成'ubuntu' 'windows' 'mac'



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



## While

```bash
#! /bin/bash

#sytax 
while [condition] 
do 
    command1
    comand2
    comand3
done


n=1 
while [ $n -le 10 ] #or (( $n <= 10 ))
do 
    echo $n 
    n=$(( $n + 1 )) # n= 不能有空格
    # or  let n++ or (( n++ ))
done 


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



## Until Loop



```bash
#! /bin/bash

#syntax
until [condition]
do 
    command1
    command2
    command3
done

n=1
until [ $n -ge 10 ] #or use (( $n > 10 ))
do 
    echo $n
    n=$(( n + 1 )) #or use (( n++ ))
done 


```



## For Loop



```bash
#! /bin/bash

for i in 1 2 3 4 5 #number separated by spaces 
do 
    echo $i
done 


for i in {1..10} #number from 1 to 10
do 
    echo $i
done 

#can only use for bash over 4.0
for i in {1..10..2} #number from 1 to 10, increase by 2
do 
    echo $i
done 


for (( i=0; i<10; i++ ))
do 
    echo $i
done 



```


![](/img/post/shell/forloop.png)

![](/img/post/shell/forloop2.png)



## For loop to execute command



```bash
#! /bin/bash

for command in ls pwd date #execute command one by one
do 
    echo "----------$command-----------"
    $command 
done 

#print all the directory inside current folder
for item in * #* every item in the directory
do 
    if [ -d $item ]  #if it is directory,  if [ -f $item ] if it is file
    then
        echo $item 
    fi 
done 


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




## Break / Continue



```bash
#! /bin/bash

for (( i=1 ; i<=10 ; i++ ))
do 
    if [ $i -gt 5 ]
    then 
        break #break the loop 当 i > 5 
    fi
    echo "$i"
done



for (( i=1 ; i<=10 ; i++ ))
do 
    if [ $i -eq 3 -o $i -eq 6 ]
    then 
        continue #continue if i = 3 or i = 6,  不print
    fi
    echo "$i"
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



Method 2 add x at the end of bash path

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


Method 3 Set x in the file

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







