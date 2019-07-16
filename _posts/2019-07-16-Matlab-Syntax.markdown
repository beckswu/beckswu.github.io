---
layout:     post
title:      "Matlab Syntax"
subtitle:   "Matlab Syntax Lookup Guide"
date:       2019-07-16 20:00:00
author:     "Becks"
header-img: "img/post-bg3.jpg"
catalog:    true
tags:
    - Matlab
    - Learning Note
---


## Container

#### Categorical

```matlab
categoryVar = categorical({'red';'yellow';'blue';'violet';'';'ultraviolet';'orange'});

```

#### Cell

```matlab
%intialize cell 
models = {'a', 'b', 'c'};

model(1) %print 'a'

```


#### Cell Array

A cell array is a data type with indexed data containers called cells. 每一个cell 可以是任意type of data. Cell arrays commonly contain pieces of text, combinations of text and numbers from spreadsheets or text files, or numeric arrays of different sizes.

There are two ways to refer to the elements of a cell array. Enclose indices in smooth parentheses, (), to refer to sets of cells — for example, to define a subset of the array. Enclose indices in curly braces, {}, to refer to the text, numbers, or other data within individual cells.

**Index**: () 是access 具体的cell array, e.g. C(1,:) ,  或者cell() e.g. C(1,2), 返回的是\[3\] ; {} 是 用来access cell content, 比如C(1,2), 返回的是3 (而不是cell). <br/>
更改cell时候, 用() 要加上cell, e.g. C(2,3) = {3} ; 用{} 不用加上cell 定义,  C{2,3} = 3

```matlab
%Create Cell Array
myCell = {1, 2, 3;
          'text', rand(5,10,2), {11; 22; 33}}
          
%myCell = 2x3 cell array
%    {[   1]}    {[          2]}    {[     3]}
%    {'text'}    {5x10x2 double}    {3x1 cell}

%Create empty Cell Array
emptyCell = cell(3,4,2); 
%emptyCell is a 3-by-4-by-2 cell array, where each cell contains an empty array, [].



%Access Cell Array

C = {'one', 'two', 'three'; 
     1, 2, 3}
    
%{
C = 2x3 cell array
{'one'}    {'two'}    {'three'}
{[  1]}    {[  2]}    {[    3]}
%}

%Cell Indexing with Smooth Parentheses, ()
upperLeft = C(1:2,1:2)

%{
upperLeft = 2x2 cell array
    {'one'}    {'two'}
    {[  1]}    {[  2]}
%}

%更新cell
C(1,1:3) = {'first','second','third'}
%C = 2x3 cell array
%    {'first'}    {'second'}    {'third'}
%    {[    1]}    {[     2]}    {[    3]}

%Convert Cell to matrix
numericVector = cell2mat(C(2,:))

%numericVector = 1×3
%     1     2     3





%Content Indexing with Curly Braces, {}
%Access the contents of cells--the numbers, text, or other data within the cells--by indexing with curly braces
C{2,3} = 300
%C = 2x3 cell array
%    {'first'}    {'second'}    {'third'}
%    {[    1]}    {[     2]}    {[  300]}


[r1c1, r2c1, r1c2, r2c2] = C{1:2,1:2} 

% r1c1 =  first'
% r2c1 = 1
% r1c2 = 'second'
% r2c2 = 2


%Concatenate the contents of the second row into a numeric array. 让cell 变成array

nums = [C{2,:}]

%nums = 1×3
%    1     2   3
```

#### Dataset

Statistics and Machine Learning Toolbox has **Dataset arrays** for storing variables with heterogeneous data types. For example, you can combine numeric data, logical data, cell arrays of character vectors, and categorical arrays in one dataset array variable.Within a dataset array, 每一个列必须是同一数据类型, but 不同的列可以是不同的数据类型. A dataset array is usually interpreted as a set of variables measured on many units of observation. That is, 每一行是一个observation, 每一列是一种variable. In this sense, a dataset array organizes data like a typical spreadsheet.

Dataset = is deep copy, if  ds1 = ds2, 如果对ds1 进行更改，不会影响ds2

```matlab 

%Read from txt
ds = dataset('File',fullfile(matlabroot,'mysubfolder','myfile.csv'),'Delimiter',',')
%fullfile returns a character vector containing the full path to the file.  f = fullfile('myfolder','mysubfolder','myfile.m')
    
%read from xlsx
ds = dataset('XLSFILE','filename', 'sheet','SheetName','range','A2:E14');
%filename,SheetName 是char array

%对每一列数据总结
summary(ds)

ds.Properties
%返回下面数据
%  Description: ''
%  VarDescription: {}
%  Units: {}
%  DimNames: {'Observations'  'Variables'}
%  UserData: []
%  ObsNames: {14x1 cell}
%  VarNames: {'sex'  'age'  'wgt'  'smoke'}

%改变DimNames
ds.Properties.DimNames{1} = 'LastName';


%Index into dataset array.


%get One column 
ds.ID  %size 是 n * 1
ds.ID(1)



%改变unique identifier  Then, delete the variable id from the dataset array.

ds.Properties.ObsNames = ds.id;
ds.id = []
```


**Clean DataSet**

TF = ismissing(A) returns a logical array (0 和 1) that indicates which elements of an array or table contain missing values. The size of TF is the same as the size of A. 比如dataset size是 5*8， 返回的也是5*8

```matlab

ds =  dataset('File',fullfile(matlabroot,'mysubfolder','myfile.csv'),'Delimiter',',')


%去除空行的，比如有的行是nan的


%方法1：
idx = isnan(ds.ModelID);
%isnan返回的是 array, size 是 ds 行数 * 1, 
params(ds,:) =[]; 
%params(ds,:) 只返回空行的, 然后让空行的为空




```


## char 

```matlab

%concatentate char array 
Str = 'abc';
['data/' abc '.mat']

```




## Write File 

```matlab


```


![](/img/post/VSCode/File_Icon_Theme.png)



