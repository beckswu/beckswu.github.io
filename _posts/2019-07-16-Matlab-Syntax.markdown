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


%Access
ds(:,2) %取第二列的数据
ds(1,:) %取所有第一行的数据
ds(1,2) %取所有第一行第二个的数据


%Get Subset from dataset
params(strcmpi(ds.Name, 'JOHN'),:); %取所有Name = "JOHN"的数据,  返回所有name = John所有行和列
%strcmpi(ds.Name, 'JOHN') 是对比每个Name的column, 返回array size = ds.Name, if == John, 那个位置返回1,else, 该位置为0


```


**Clean DataSet**

TF = ismissing(A, indicator) returns a logical array (0 和 1) that indicates missing values. TF size is the same as the size of A. 比如dataset size是 5\*8， 返回的也是5\*8. Indicator, missing value indicators. 如果A is array, then indicator 必须是vector. If A table, indicator 可以是cell arra with entires of multiple datatypes. 如果想add indicators while maintaining the list of standard indicators, must include all default indicators. 例如, A is table wtih categorical and numeric values, use ismissing(A,{-99,'<undefined>'}) to indicate -99 as a missing numeric value, but preserve <undefined> as a missing categorical value.
    
```matlab
%{
A=5×4 table
    dblVar    int8Var    cellstrVar    charVar
    ______    _______    __________    _______

     NaN          1       'one'           A   
       3          3       'three'         C   
     Inf          5       ''              E   
       7          7       'NA'                
       9        -99       'nine'          I   
%}

id = {'NA' '' -99 NaN Inf};
TF = ismissing(A,id)

```

```matlab

ds =  dataset('File',fullfile(matlabroot,'mysubfolder','myfile.csv'),'Delimiter',',')


%去除空行的，比如有的行是nan的


%方法1：
idx = isnan(ds.ModelID);
%isnan返回的是 array, size 是 ds 行数 * 1, 
params(ds,:) =[]; 
%params(ds,:) 只返回空行的, 然后让空行的为空

%方法二
id =ismissing(params,'NumericTreatAsMissing',-99,...
                 'StringTreatAsMissing',{'NaN','.','NA'});


%Determine if any array elements are nonzero, dim = 2 表示对每一行操作，dim = 1表示对列进行check
params(any(ix,2),:) = []


```

**Export DataSet**

```matlab

%To Txt File 
% Export the dataset array, a text file named ds.txt. By default, export writes to a tab-delimited text file
% 会create ds.txt in working folder, 如果file 已经存在, overwrites the existing file. 
export(ds) %ds is dataset


%Export without Column Name
%Export ds with variable names suppressed to a text file named NoLabels.txt.
export(ds,'File','NoLabels.txt','WriteVarNames',false)
%There are no variable names in the first line of the created text file, NoLabels.txt.

%Export to a comma-delimited format.
export(ds,'File','Fun.csv','Delimiter',',')


%Export to an Excel spreadsheet.
export(ds,'XLSFile','Fun.xlsx')

```


#### Struct

A structure array is a data type that groups related data using data containers called fields. Each field can contain any type of data. 
每一个field 相当与cell, type可以不一样

s = struct(field,value) creates a structure array with the specified field and value. 
- If value 不是 cell array, or if value is a scalar cell array, then s is a scalar structure(key-value形式，而不是key-column of values形式). For instance, ``` s = struct('a',\[1 2 3\]) ``` creates a 1-by-1 structure, where s.a = \[1 2 3\].
- If value is a nonscalar cell array, then s is a structure array with the same dimensions as value(与value dim 一样). Each element of s contains the corresponding element of value. For example, s = struct('x',{'a','b'}) returns s(1).x = 'a' and s(2).x = 'b'.
- If value is an empty cell array {}, then s is an empty (0-by-0) structure. e.g. ``` s = struct('a',{},'b',{},'c',{}) ```

s = struct(field1,value1,...,fieldN,valueN) creates a structure array with multiple fields.
- 如果value都不是cell array or all values 在cell array里是scaler, s 是 scaler 
- If 任何value是 nonscalar cell array, then s 是cell array. 
    - 保持scaler和不是scaller的dimension 一样 to insert content of value in that field for all element of s , E.g. s = struct('x',{'a','b'},'y','c') returns s(1).x = 'a', s(2).x = 'b', s(1).y = 'c', and s(2).y ='c'
    - 两个不是scaler 的cell array dimension不能不一样, 比如```Params = struct('name', cell(3,1), 'sex', {0}, 'id', cell(2,1));``` 会报错, ``` Params = struct('name', cell(3,1), 'sex', {0}, 'id', cell(3,1));``` 这样才可以, 但是每个cell array可以放任意的东西
    
    


```matlab

%创建struct
%方法一
s.a = 1;
s.b = {'A','B',2}

%方法二 Structure with One Field
field = 'f';
value = {'some text';
         [10, 20, 30];
         magic(5)};  %value 需要时cell array的形式
s = struct(field,value)
s.f 

%{
ans = 
'some text'
ans = 1×3

    10    20    30

ans = 5×5

    17    24     1     8    15
    23     5     7    14    16
     4     6    13    20    22
    10    12    19    21     3
    11    18    25     2     9
%}


%方法三 Structure with Multiple Fields
Params = struct('name', cell(3,1), 'beta', cell(3,1), 'ref', cell(3,1));
s = struct('a',{},'b',{},'c',{})


%struct with Empty Field
s = struct('f1','a','f2',[])
%s = struct with fields:
%    f1: 'a'
%    f2: []

%index 
s(1) %返回第一行所有数据
s(1).a %只要第一行a的数据


```


#### char 

```matlab

%concatentate char array 
Str = 'abc';
['data/' abc '.mat']

```


#### Convert Data

```matlab

messyData.var2 = str2double(messyData.var2); %把table的str data convert to double

```

## Save 

```matlab
%把variable ds save 进mat file
save(['data.mat'],'ds'); %如果不加ds 会存入所有working directory的variable 
```


## Write File 

```matlab

fid = fopen(outputFileName, 'w');
fprintf(fid, ',%.16f', ds(i).id);
%比如ds(i).id是matrix/array, fprint 会打印所有的variable 用逗号隔开, like ,1,2,3,4,5

fclose(fid);%关掉这在写的file


```


![](/img/post/VSCode/File_Icon_Theme.png)



