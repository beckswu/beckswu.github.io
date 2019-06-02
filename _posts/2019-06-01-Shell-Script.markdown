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


**cp**: copy and paste  <br/>
**mv**: move and rename <br/>
**Less**:  View some part of the file (lookup, view line by line / page by page) <br/>
**Touch**: create new file(cannot create directory), change file timestamp   <br/>
**Nano(or gedit)**: txt/code editor   <br/>
**sudo**: grant super user priviledge <br/>
**Top**: provide you dynamic real time view of running system <br/>
**Echo** print, can include variable in "" <br/>



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
#! bin/bash  #intepreter know it is bash script
echo "Hello world" #print Hello world
```

## variable

```shell

```


## cd

**home directory和root directory** 不一样, root是/, home 是 /Users/ username 的文件夹

```shell
cd / #go to root directory
cd ~ #home directory
cd .. #到parent directory
cd Documents #go to Document Directory
cd /home/programming/Documents/ #功能与上面一样, Go to Document Directory
cd My\ Books # go to My Books folder, 在My Books中间有空格
cd "My Books" #功能与上面一样
cd 'My Books' #功能与上面一样

```



## cat

cat: 1. Display Txt 2. combine Txt file 3. Create new Txt file

syntax: cat options file1 file2 ... 

```shell
cat Hello World #会打印 echo Hello world
#Ctrl D means the end of cat command 

cat list1.txt #显示list1.txt 所有内容
cat list1.txt list2.txt #显示list1.txt 和list2.txt所有内容，先显示list1的再显示list2的

cat -b list1.txt #把list1.txt的 不是blank的line(空行) 显示序号
cat -n list1.txt #把list1.txt的 所有行(空行或者不空行) 都显示序号
cat -s list1.txt #squeeze 连续 blank line to one blank line

cat -E list1.txt #add $ at the end of each line

man cat #显示cat所有function

cat > test.txt  #把接下来input的内容 output 到test.txt，输完了 按Ctrl+D， test.txt之前内容被remove
cat >> text.txt #把接下来input的内容 append 到test.txt

cat list1.txt list2.txt > out.txt #把mlist1, list2的内容合并，生成out.txt
cat list1.txt list2.txt > list2.txt #这样是不行的，不能把input 当成output file
cat list1.txt >> list2.txt #修改上面一行的error，append list1.txt 到list2.txt

```




## mkdir

```shell
mkdir image #生成image directory
mkdir image/pic #生成pic directory inside image directory
mkdir names/mark #当names 不在当前文件夹下，显示error, No such file or directory
mkdir -p names/mark (mkdir --parents names/mark ) #-p means -- parents
mkdir -p names/{john,tom,bob}  #creat several directory inside current directory, 建立三个文件夹,john, tom,bob
# {john,tom,bob} 不能有空格，否则建立的文件夹是 {john,


```



## rm & rmdir

**rmdir**: remove directory, **rm**: rmove file and directory

```shell

rmdir abc # remove abc的folder
rmdir a/b/c/d/e #只remove 最后e的directory
rmdir -p a/b/c/d/e  #remove 所有的directory structure
rmdir -pv a/b/c/d/e #remove 所有directory structure，并显示(verb)remove的进程
#如果a/b/c/d/e 每个并不是空的文件夹，会显示error, failed to remove directory a/b: Directory not empty

rm -rv a/b 
rm -rv a #与上面一行作用是一样的

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
![](/img/post/linux/pic1.png)


![](/img/post/linux/watch.png)







