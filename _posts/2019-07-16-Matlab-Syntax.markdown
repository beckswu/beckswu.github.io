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

#### Matrices

Index
    - 可以传入list of indexs 然后根据column的index, 返回数
    - ```X(:,1)``` 获取所有第一列的数, ```X(1,:)```获取第一行的数, 不可以```x(:2,1)```
    - X(1,2) 获取第二行第三个数

Creat
    - `x = 0:2:8`: 生成一个array, 从0开始到8, 每次跳2 ```0 2 4 5 6 8```

```

x = [1 2 3; 4 5 6; 7,8 9]

%{ x =
     1     2     3
     4     5     6
     7     8     9 %}
     
x([1,2,3,7,8])
%{ ans = 
    1     4     7     3     6     %}


x([1;2;3;7;8])
%{ans =

     1
     4
     7
     3
     6 %}
    
    

```

#### Struct

A structure array is a data type that groups related data using data containers called **fields**. Each field can contain any type of data. 每一个field 相当与cell array, type可以不一样

s = struct(field,value) creates a structure array with the specified field and value. 
- If value 不是 cell array, or if value is a scalar cell array(e.g. 一个数 or matrix), then s is a scalar structure, For instance, ``` s = struct('a',[1 2 3]) ``` creates a 1-by-1 structure, where ```s.a = [1 2 3]```.
- If value is a nonscalar cell array, For example, s = struct('x',{'a','b'}) returns s(1).x = 'a' and s(2).x = 'b'.
- If value is an empty cell array {}, then s is an empty (0-by-0) structure. e.g. ``` s = struct('a',{},'b',{},'c',{}) ```

s = struct(field1,value1,...,fieldN,valueN) creates a structure array with multiple fields.
- 如果value都不是cell array or all values 在cell array里是scaler, s 是 scaler 
- If 任何value是 nonscalar cell array, then s 是cell array. 
    - Dimension 与nonscalar cell array 一样, 对于scaler 会 insert content of value in that field for all element of s , E.g. s = struct('x',{'a','b'},'y','c') returns s(1).x = 'a', s(2).x = 'b', s(1).y = 'c', and s(2).y ='c'
    - 两个nonscaler 的cell array dimension 要一样, 比如```Params = struct('name', cell(3,1), 'sex', {0}, 'id', cell(2,1));``` 会报错, ``` Params = struct('name', cell(3,1), 'sex', {0}, 'id', cell(3,1));``` 这样才可以, 但是每个cell array可以放任意的东西
    
    


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


## Function 
- ```nargin```: function 有几个input, input 数量可能因为有```varargin```不同function call有变化
- ```nargout```: 看有几个output, 如果返回负数表示第i个位置返回时varargout,
- ```varargout```:Variable-length output argument list. 用于function, 可以返回any number of output arguments, include varargout as function last  output after explicitly declared output 
- ```varargin```: 让function accept 任意number of input arguments, 放在function input最后面. varagin 是 1 by N cell array. N is number of input that function receives after explicitly declared inputs. 如果function recieves no input, varargin is empty cell array
    - ```varargin{:}```: to get content of all varargin. 

```matlab
% -----------nargout ---------------

%{
比如有下面的function
function [dif,absdif] = subtract(y,x)
    dif = y-x;
    if nargout > 1
        disp('Calculating absolute value')
        absdif = abs(dif);
    end
end
%} 

fun = @subtract;
nargout(fun)
% print ans = 2



%nargout can determine how many outputs a function that uses varargout can return.
%{
function [sizeVector,varargout] = mySize(x)
    sizeVector = size(x);
    varargout = cell(1,nargout-1);
    for k = 1:length(varargout)
        varargout{k} = sizeVector(k);
    end
end
%}

fun = 'mySize';
nargout(fun)
%print ans = -2; 
% The minus sign indicates that the second output is varargout. The mySize function can return an indeterminate number of additional outputs.



% -----------varargout ---------------


function [s,varargout] = returnVariableNumOutputs(x)
    nout = max(nargout,1) - 1;
    s = size(x);
    for k = 1:nout
        varargout{k} = s(k);
    end
end

A = rand(4,5,2);
[s,rows,cols] = returnVariableNumOutputs(A) %在function里 nargout是3
% print s = 1×3
%  4     5     2
% rows = 4
% cols = 5



A = zeros(1,4,5,2);
[s,dim1,dim2,dim3] = returnVariableNumOutputs(A) 在function里 nargout是4
% print s = 1×4 
%   1     4     5     2
% dim1 = 1
% dim2 = 4
% dim3 = 5




```

#### Built-in Method

**find**

```k = find(X)``` returns a vector containing the linear indices of each nonzero element in array X. 找X的非0 element. ```k2 = find(~X)``` to locate zeros in X
- If X is a vector, then find returns a vector with the same orientation as X.
- If X 是 multidimensional array, returns a column vector of the linear indices of the result.
- If X contains a nonzero elements or is empty, return empty array()

```matlab
X = [1 0 2; 0 1 1; 0 0 4]

%{
X = 3×3

     1     0     2
     0     1     1
     0     0     4
%}

k = find(X)
%{  k = 5×1
     1
     5
     7
     8
     9       %}
   
%Use the logical not operator on X to locate the zeros.
k2 = find(~X)
     
```

```k = find(X,n)``` returns the first n indices corresponding to the nonzero elements in X.

```matlab
X = magic(4)
%{ X = 4×4
    16     2     3    13
     5    11    10     8
     9     7     6    12
     4    14    15     1 %}

k = find(X<10,5)
%{ k = 5×1
     2
     3
     4
     5
     7 %}
     
X(k)
%{ ans = 5×1
     5
     9
     4
     2
     7
%}


% -----------varargin --------------

function varargout = redplot(varargin)
    %disp(varargin);
    %varargin{:}
    [varargout{1:nargout}] = plot(varargin{:},'Color',[1,0,0]);
end

x = 0:pi/100:2*pi;
y = sin(x);
redplot(x,y)
h = redplot(x,y,'Marker','o','MarkerEdgeColor','green'); 
```


#### Function Handle 

Function Handle is datatype 用于储存 an association to a function. 用于
- pass a function into another function. e.g. pass function to integration / optimization functions
- specifiy callback functions (UI)
- Construct handles to function defined inline instead of stored in program file (anonymous functions)
- Call local function from outside the main function 


**Create Function Handle**

```matlab 

function y = computeSquare(x)
y = x.^2;
end

f = @computeSquare;
a = 4;
b = f(a)

% 如果function 不需要input, 用empty parentheses
h = @ones;
a = h(); %print a = 1

%without parentheses, the assignment creates another function handles 
a = h % print a = @ones

```

**Anonymous Functions**

Anonymous Functions is one line expression which does not require a program file. 用comma separate list as the input arguments to the anoymous function. ```h = @(arglist)anonymous_function```

```matlab

sqr = @(n) n.^2
x = sqr(3) % print x = 9

sq = @(a,b,c) [a.*2,b.*2,c.*2]
sq(1,2,3) % print ans =  2, 4, 6


```


**Arrays / Structure Array of Function Handles**

```matlab
C = {@sin, @cos, @tan};
C{2}(pi) % print ans = -1

S.a = @sin;  S.b = @cos;  S.c = @tan;
S.a(pi/2) % print ans = 1

```

## Save 

```matlab
%把variable ds save 进mat file
save(['data.mat'],'ds'); %如果不加ds 会存入所有working directory的variable 
```


## Print/ Write File 

```matlab


formatSpec = "Size of varargin cell array: %dx%d";
str = compose(formatSpec,size(varargin));
disp(str)


fid = fopen(outputFileName, 'w');
fprintf(fid, ',%.16f', ds(i).id);
%比如ds(i).id是matrix/array, fprint 会打印所有的variable 用逗号隔开, like ,1,2,3,4,5

fclose(fid);%关掉这在写的file


```



![](/img/post/VSCode/File_Icon_Theme.png)



