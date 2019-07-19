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

## Class

#### Abstract Class

Class is Abstract classes if(满足任意一个,即是)
- Abstract class attribute
    - Concrete subclasses must redefine any properties or methods that are declared as abstract.
    - The abstract class does not need to define any abstract methods or properties.
    - 可以定义nonabstract method or class
- has abstract method 
    - do not define abstract method(<span style="color:red">no implementation</span>), use only method signature
    - Concrete subclasses <span style="background-color:#FFFF00">are not required to have same number of input  output arguments and do not need to have same argument name. 只需the same signuature </span>
- has abstract property 
    - Concrete subclasses must redefine abstract properties without the Abstract attribute.
    - Concrete subclasses must use the same values for the SetAccess and GetAccess attributes as abstract superclass.
    - Abstract properties cannot define access methods and cannot specify initial values(不要定义取值method). The subclass can create access methods and specify initial values.
	
	

Abstract classes 
- can defeine properties and methods that are not abstract (可以定义非abstract properites, method 继承到subclass)
- Pass on their concrete members through inheritance
- Do not need to define any abstract members	


```matlab

% abstract class attribute
classdef (Abstract) AbsClass 
   ...
end

classdef (Abstract) AbsClass 
   methods (Abstract)
     abstMethod(obj) %abstract method 
   end
   methods 
     abstMethod(obj) %non abstract method 
   end
end



% Abstract Methods

methods (Abstract)
   abstMethod(obj)
end

% abstract properties

properties (Abstract)
   AbsProp
end

```


**Implementing a Concrete Subclass**

A subclass <span style="background-color:#FFFF00">must implement all inherited abstract properties and methods</span> to become a **concrete class**, 否则the subclass is an abstract class. 

Matlab does not force subclasses to implement concrete methods with the same signature or attributes


Determine if a class is abstract 

```matlab 

classdef AbsClass
   methods(Abstract)
      result = absMethodOne(obj)
      output = absMethodTwo(obj)
   end
end

mc = ?AbsClass;
if ~mc.Abstract
   % not an abstract class
end

```

Display Abstract Member Names
```matlab 
meta.abstractDetails('AbsClass');
%print 
Abstract methods for class AbsClass:
   absMethodTwo   % defined in AbsClass
   absMethodOne   % defined in AbsClass
   
 
 
classdef SubAbsClass < AbsClass
% Does not implement absMethodOne
% defined as abstract in AbsClass
   methods
      function out = absMethodTwo(obj)
         ...
      end
   end
end


meta.abstractDetails(?SubAbsClass)
Abstract methods for class SubAbsClass:
   absMethodOne   % defined in AbsClass
The SubAbsClass class is abstract because it has not implemented the absMethodOne method defined in AbsClass.

msub = ?SubAbsClass;
msub.Abstract
ans =

     1
   
```



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


#### char 

```matlab

%concatentate char array 
Str = 'abc';
['data/' abc '.mat']

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
    - ```end```: 用end表示到结束, X(2:end), 从第二位开始到结束

Creat
    - `x = 0:2:8`: 生成一个array, 从0开始到8, 每次跳2 ```0 2 4 5 6 8```

```
y = [1:4]
% y = [1 2 3 4]

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


#### Table 

Table array with named variable that contain different types 

table arrays store column-oriented or tabular data, such as columns from a text file or spreadsheet. Table store each piece of column-oriented data in a *variable*. <span style="background-color:#FFFF00">Table varaibles can have different data type</span> 但是row 行数需要一样. Use ```summary``` function to get information about a table 

**Index**: use smooth parentheses ```()``` to return a subtable or curly braces ```{}``` to extract the contents. <br/>
- access column ```T.Age```
	- access single value content from column ```T.Age(1)``` or ```T{1,1}```, 但不可以```T.Age{1}```, 因为T.Age已经是matrix了
	- access single value in column ```T(1,1)```, 返回有header和row的名字
- Get all data without header/row names ``` T{:,:}.``` is the same as ```T.Variables```
- Get Row Data 
	- Get row data content and return in arrary or vector 

**Creation**

- ```T = table(var1,...,varN)``` Variables can be of different sizes and data types, but all variables 必须#rows 一样. Example: ```table([1:3]',{'one';'two';'three'},categorical({'A';'B';'C'}))```, Common input variables are <span style="color:red">numeric arrays, logical arrays, character arrays, structure arrays, or cell arrays</span>
- ```T = table('Size',sz,'VariableTypes',varTypes)``` preallocates space for the variables that have data types you specify. sz is a two-element numeric array, where ```sz[1]``` specifies *#rows* and ```sz[2]``` *# variables*. varTypes is a cell array of character vectors specifying data types. ``` T = table('Size',[50 3],'VariableTypes',{'string','double','datetime'})```
- ```T = table(___,'VariableNames',varNames)``` specifies the names of the variables in the output table.  ```T = table(categorical({'M';'F';'M'}),[45;32;34], {'NY';'CA';'MA'},logical([1;0;0]), 'VariableNames',{'Gender','Age','State','Vote'})```
- ```T = table(___,'RowNames',rowNames) ``` specifies names of the rows in table, ```T = table(Age,Weight,Height,'RowNames',LastName)``` ```T = table([10;20;30],{'M';'F';'F'},'VariableNames',{'Age','Gender'},'RowNames',{'P1','P2','P3'})```

```matlab
sz = [4 3];
varTypes = {'double','datetime','string'};
varNames = {'Temperature','Time','Station'};
T2 = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames)
T2=4×3 table
    Temperature    Time     Station 
    ___________    ____    _________

         0         NaT     <missing>
         0         NaT     <missing>
         0         NaT     <missing>
         0         NaT     <missing>
	 
%Build Table by assigning variables individually	 
Date = {'12/25/11','1/2/12','1/23/12','2/7/12','2/15/12'};
location1 = [20 5 13 0 17];
location2 = [18 9 21 5 12];
location3 = [26 10 16 3 15];

T = table;
T.Date = Date';
T.Natick = location1';
T.Boston = location2';
T.Worcester = location3'	

T=5×4 table
       Date       Natick    Boston    Worcester
    __________    ______    ______    _________

    '12/25/11'      20        18         26    
    '1/2/12'         5         9         10    
    '1/23/12'       13        21         16    
    '2/7/12'         0         5          3    
    '2/15/12'       17        12         15    
```

**Table Operation**

- ```mean(T.ColumnName)```: to calculate mean for one column , ```meanHeight = mean(T.Height)```
- Create new column/variable: ```T.BMI = (T.Weight*0.453592)./(T.Height*0.0254).^2``` BMI是new column
- **Adding Row**
	- by Concatenate: ```T =  [T;{50,60,70}]``` can directly concatenate from cell array
	- by from struct 
- **Omit Duplicate Rows**: ```Tnew = unique(Tnew); ```	
- **Delete Rows**
	- **by row number**: ```Tnew([18,20,21],:) = [];```
	- **by Row Name**: ```Tnew('Smith',:) = [];```
	- **Search Row to delete**: ```toDelete = Tnew.Age < 30; Tnew(toDelete,:) = [];```
- **Number of Rows** : ```H = height(T)``
- **Number of Columns**: ```H = width(T)``


Adding row 
```matlab

LastName = {'Sanchez';'Johnson';'Lee';'Diaz';'Brown'};
Age = [38;43;38;40;49];
Height = [71;69;64;67;64];
Weight = [176;163;131;133;119];
T = table(Age,Weight,Height,'RowNames',LastName)

struct.Age = 50;
struct.Weight = 60;
struct.Height = 70;
T = [T;struct2table(struct)]


```


**Sort**
- ```B = sortrows(A)``` sorts the rows of a matrix in ascending(由小到大) order based on the elements in the first column(是第一列不是index的column). 如果第一列有repeat elements, 会根据第二列再比较, 以此列推
- ```B = sortrows(A,column)``` sorts A based on specified column. ```sortrows(A,4)``` 根据第四列sort,  ```sortrows(A,[4 6])```根据第四列sort, 有tie的话再根据第六列
	- **sort based on column name**: ```sortrows(T,'Age')```, 根据age column 进行sort, ```sortrows(tblA,{'Height','Weight'},{'ascend','descend'})``` 先根据Height 由小到大排列, 再根据Weight 由大到小排列
	- ```sortrows(A,[4 6],{'ascend' 'descend'})```, 先根据第四列由小到大排列, 再根据第6列由大到小排列,
- ```B = sortrows(___,Name,Value)``` specifies additional parameters for sorting rows. For example, ```sortrows(A,'ComparisonMethod','abs')``` sorts the elements of A by magnitude.
- ```[B,index] = sortrows(___)``` returns an index vector that describes the rearrangement of rows ```[E,index] = sortrows(A,4,'descend')``` 返回sort之后1到n行的 属于sort 之前的 index
- ``` sortrows(T,'RowNames')``` sort table based on index column
- ```tblB = sortrows(___,Name,Value)``` specifies additional parameters for sorting rows of a table or timetable. ```sortrows(tblA,'Var1','MissingPlacement','first')``` sorts based on the elements in Var1, ordering missing elements such as NaN at the beginning of the table. 把Missing element放最前面


**issortedrows** 跟sort syntax 类似, return 1(true) 满足sort 条件, otherwise return 0 (false)

- ```TF = issortedrows(A)``` returns if first column of a matrix or table A are listed in ascending order, 
- ```TF = issortedrows(A,column) ``` returns sorted based on 特定的columns, e.g. ```issortedrows(A,[4 6])```
	- ```issortedrows(A,[2 3],{'ascend' 'descend'})```: check if A 是按照先第2列由小到大排列, 再按照第3列由大到小排列
	- ```issortedrows(tblA,{'Age','Weight'})```: check if 先按照Age 再按照Weight sort的
- ```TF = issortedrows(___,Name,Value)```: ```issortedrows(A,'ComparisonMethod','abs')``` checks if the elements in the first column of A are sorted by magnitude.
- ```TF = issortedrows(tblA,'RowNames')``` TF = issortedrows(tblA,'RowNames') 是不是按照Table index row进行sort
- ```TF = issortedrows(___,Name,Value)``` specifies additional parameters for sorting tables. ```issortedrows(tblA,'Var1','MissingPlacement','first') ```checks that missing elements in Var1, such as NaN or NaT, are placed at the beginning of the table.



issortedrows的direction
- 'ascend' (default) — 由小到大. Data can contain consecutive repeated elements.
- 'descend' — 由大到小. Data can contain consecutive repeated elements.
- 'monotonic' — 由小到大 or 由大到小 . Data can contain consecutive repeated elements.
- 'strictascend' —  由小到大没有重复. Data cannot contain duplicate or missing elements.
- 'strictdescend' — 由大到小没有重复. Data cannot contain duplicate or missing elements.
- 'strictmonotonic' —由小到大没有重复 or 由大到小没有重复. Data cannot contain duplicate or missing elements.

```matlab 
%Complex Matrix

A = [1+i 2i; 1+2i 3+4i]
A = 2×2 complex
   1.0000 + 1.0000i   0.0000 + 2.0000i
   1.0000 + 2.0000i   3.0000 + 4.0000i

TF = issortedrows(A,'ComparisonMethod','real')
%F = logical
%   1


```

**Properties**

return a summary of all the metadata properties using the syntax ```tableName.Properties.```

- **DimensionNames**: 1 by 2 cell array，default value是  ````{'Row'} {'Variables'}``` 更改会影响index, 比如Variable 改为Data, 那么直接access所有数据是```T.Variables``` 变成 ```T.Data```
- **RowNames**: 相当于每行的index,  specified as a **cell array** of character vectors or a string array, whose <span style="color:red">elements are nonempty and distinct</span>. Matlab remove了 any leadning or trailing white space from the row names. Row names are visible when you view the table. 可以用T.Row access Row Index Value/Content
- **Discription**: 更改Description, 会在**summary**中显示, <span style="background-color：#FFFF00">summary 显示每一列数的Min, Median, Max </span>
- **VariableNames**: a cell array of char vectors or string array,  whose <span style="color:red">elements are nonempty and distinct</span>. 如果不specify variable name or specify invalid identifier, MATLAB 自动生成'Var1' ... 'VarN' Where N is #variables
- **VariableDescription**：辅助信息，显示在当用```summary```时候, default an empty cell array. 如果不为空，size = #variables. 可以specifiy empty char vector or empty string 对于没有description的variable
- **VariableUnits**: 辅助信息，显示在当用```summary```时候

DimensionNames
```matlab 

load patients
T = table(Age,Height,Weight,Systolic,Diastolic, ...
          'RowNames',LastName);
T.Properties.DimensionNames

ans = 1x2 cell array
    {'Row'}    {'Variables'}


T.Properties.DimensionNames = {'Patient','Data'};
T.Properties

ans = 
  TableProperties with properties:
             Description: ''
                UserData: []
          DimensionNames: {'Patient'  'Data'}
           VariableNames: {'Age'  'Height'  'Weight'  'Systolic'  'Diastolic'}
    VariableDescriptions: {}
           VariableUnits: {}
      VariableContinuity: []
                RowNames: {100x1 cell}
        CustomProperties: No custom properties are set.
      Use addprop and rmprop to modify CustomProperties.

```

RowNames
```matlab

load patients
T = table(Gender,Age,Height,Weight,Smoker,Systolic,Diastolic);

T.Properties.RowNames = LastName;
head(T,4)
ans=4×7 table
                 Gender     Age    Height    Weight    Smoker    Systolic    Diastolic
                ________    ___    ______    ______    ______    ________    _________
    Smith       'Male'      38       71       176      true        124          93    
    Johnson     'Male'      43       69       163      false       109          77    
    Williams    'Female'    38       64       131      false       125          83    
    Jones       'Female'    40       67       133      false       117          75 

T.Properties.DimensionNames
ans = 1x2 cell array
    {'Row'}    {'Variables'}

T.Row(1:5)
ans = 5x1 cell array
    {'Smith'   }
    {'Johnson' }
    {'Williams'}
    {'Jones'   }
    {'Brown'   }
    
T({'Smith','Williams'},:)
ans=2×7 table
                 Gender     Age    Height    Weight    Smoker    Systolic    Diastolic
                ________    ___    ______    ______    ______    ________    _________

    Smith       'Male'      38       71       176      true        124          93    
    Williams    'Female'    38       64       131      false       125          83    
    
```

Description/Summary

```matlab 

load patients
T = table(Gender,Age,Height,Weight);
T.Properties.Description = 'Simulated patient data';
summary(T)

Description:  Simulated patient data
Variables:
    Gender: 100x1 cell array of character vectors
    Age: 100x1 double
        Values:
            Min        25  
            Median     39  
            Max        50  

```

VariableNames
```matlab 

T = table({'M';'M'},[38;43], [71;69],[176;163])
T=5×4 table
    Var1    Var2    Var3    Var4
    ____    ____    ____    ____
    'M'      38      71     176 
    'M'      43      69     163 

T.Properties.VariableNames = {'Gender','Age','Height','Weight'}
T=5×4 table
    Gender    Age    Height    Weight
    ______    ___    ______    ______
     'M'      38       71       176  
     'M'      43       69       163  

```
VariableDescriptions / VariableUnits

```matlab
load patients
T = table(Gender,Age,Height,Weight,Smoker,Systolic,Diastolic);
T.Properties.VariableDescriptions = {'Male or Female','','','', ...
                                     'Has the patient ever been a smoker', ...
                                     'Systolic Pressure','Diastolic Pressure'};
T.Properties.VariableUnits = {'','Yrs','In','Lbs','','mm Hg','mm Hg'};
summary(T)
Variables:
    Gender: 100x1 cell array of character vectors
    Age: 100x1 double
    	Properties:
            Description:  Male or Female
        Values:
            Min        25  
            Median     39  
            Max        50  

    Height: 100x1 double
   	Properties:
            Units:  Yrs
        Values:
            Min          60   
            Median       67   
            Max          72   

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

**cd**

```oldFolder = cd(newFolder)``` returns the existing current folder to oldFolder, and then it changes the current folder to newFolder.



**sprintf** 

Format data into string or character vector

```str = sprintf(formatSpec,A1,...,An)``` formats the adata in arrays A1,.... An usign the formatting formatrSpec and return into str/char arrray. Sprintf formats the values in A1,...,An in column ordered. 如果formatSpec是str, output is str. Otherwise, str is char array

```[str,errmsg] = sprintf(formatSpec,A1,...,An)``` returns an error message as a character vector when the operation is unsuccessful. Otherwise, errmsg is empty

```matlab

A = 1/eps;
str_e = sprintf('%0.5e',A)
%print str_e =  '4.50360e+15'


formatSpec = 'The array is %dx%d.';
A1 = 2;
A2 = 3;
str = sprintf(formatSpec,A1,A2)
%str = 'The array is 2x3.'

%Specify the minimum width of the printed value.

str = sprintf('%025d',[123456]) %str = sprintf('%025d',123456) 结果是一样的
% str =  '0000000000000000000123456'


%Reorder the input value 
A1 = 'X';
A2 = 'Y';
A3 = 'Z';
formatSpec = ' %3$s %2$s %1$s';
str = sprintf(formatSpec,A1,A2,A3)
%str =  ' Z Y X'

%Create Character Vector from Values in Cell Array
%一个column 一个column的打
C = { 1,   2,   3 ;    'AA','BB','CC'};
str = sprintf(' %d %s',C{:})
% str = ' 1 AA 2 BB 3 CC'


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



