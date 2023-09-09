# 列表的数据项不需要具有相同的类型
list1 = ['Google', 'Runoob', 1997, 2000]
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]
list4 = ['red', 'green', 'blue', 'yellow', 'white', 'black']

# 元组与列表类似,不同之处在于元组的元素不能修改,元组写在小括号()里,元素之间用逗号隔开
tup1 = ('Google', 'Runoob', 1997, 2000)
tup2 = (1, 2, 3, 4, 5 )
tup3 = "a", "b", "c", "d"   #  不需要括号也可以

# 集合（set）是一个无序不重复元素的序列。基本功能是进行成员关系测试和删除重复元素
# 可以使用大括号 { } 或者 set() 函数创建集合
set1 = {1, 2, 3, 4}            # 直接使用大括号创建集合
set2 = set([4, 5, 6, 7])      # 使用 set() 函数从列表创建集合

# 字典(dictionary)是除列表以外python之中最灵活的内置数据结构类型
emptyDict = dict()
tinydict2 = { 'abc': 123, 98.6: 37 }
tinydict2['School'] = "菜鸟教程"  # 添加信息
tinydict2.clear()     # 清空字典
del tinydict2         # 删除字典

# 迭代器
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
print (next(it))
# or
for x in it:
    print (x, end=" ")

class MyClass:
    """一个简单的类实例"""
    i = 12345
    __weight = 0 # 私有变量
    def __init__(self):
        self.data = []
    def f(self):
        return 'hello world'
    def __foo(self):          # 私有方法
        print('这是私有方法')
 
# 实例化类
x = MyClass()
 
# 访问类的属性和方法
print("MyClass 类的属性 i 为：", x.i)
print("MyClass 类的方法 f 输出为：", x.f())

class ExtenedMyClass(MyClass):
    """docstring for ExtenedMyClass"""
    def __init__(self, arg):
        MyClass.__init__() # 调用父类的构函数
        self.arg = arg
    def bark(self):
        print("bark", self.i)
    def f(self):
        return 'hello world 2'

# 模块
import sys # 引入 sys 模块

print('命令行参数如下:')
for i in sys.argv:
   print(i)
 
print('\n\nPython 路径为：', sys.path, '\n')

import support
# from support import print_func
# from support import *

support.print_func("哈哈")
dir(support)

# 这里给出了一种可能的包结构（在分层的文件系统中）:
# sound/                          顶层包
#       __init__.py               初始化 sound 包
#       formats/                  文件格式转换子包
#               __init__.py
#               wavread.py
#               wavwrite.py
#               aiffread.py
#               aiffwrite.py
#               auread.py
#               auwrite.py
#               ...
#       effects/                  声音效果子包
#               __init__.py
#               echo.py
#               surround.py
#               reverse.py
#               ...
#       filters/                  filters 子包
#               __init__.py
#               equalizer.py
#               vocoder.py
#               karaoke.py
#               ...

