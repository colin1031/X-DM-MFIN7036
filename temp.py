import requests
import xlrd
from bs4 import BeautifulSoup
from xlutils.copy import copy 

 # 调用xlrd库，打开stocklist表,并查看所有表名，返回所需要的表及其行列数
wb = xlrd.open_workbook('../Desktop/stocklist.xls')
print(wb.sheet_names())

 # 查看沪A的股票列表
sheet1 = wb.sheet_by_index(0)
print("在stocklist表中共有"+str(sheet1.nrows)+"行")
print("在stocklist表中共有"+str(sheet1.ncols)+"列")

 # 定义函数，查找class为定值的div，返回子节点，并筛选有用的信息行，返回字典信息
def msg(classname):
    global soup
    st=[]

    i=1
    for child in soup.find("div",{"class":classname}).children:
        if (i%2==0 and i<24):
             # 返回指标内容
            index=str(child).split("<")[2].split(">")[1]
             # 返回数值内容
            val=str(child).split("<")[4].split(">")[1]
             # 存为字典格式
            dic={index:val}
             # 插入到列表中
            st.append(dic)
        i=i+1
    return st


 # 声明一个空列表以存储字典信息
stockmsg=[]
indexlist=["股票编号","股票名称","今开","昨收","成交量","换手率","最高","最低","涨停","跌停","内盘","外盘","成交额","振幅","委比","量比","流通市值","总市值","市盈率","市净率","每股收益","每股净资产","总股本","流通股本"]

 # 对每一个股票代码构造url
for i in range(1,sheet1.nrows):
     # 取第i+1行第1列的股票代码，调用get方法，可以修改sheet1.nrows
    num=sheet1.cell(i,0).value
    name=sheet1.cell(i,1).value
    url='https://gupiao.baidu.com/stock/sh{}.html'.format(num)
    r = requests.get(url)
    
      # 声明异常以防页面不存在
    try:
        r.raise_for_status()
        r.encoding=r.apparent_encoding
    except:
        print("爬取失败")

     # 调用BeautifulSoup库，返回渲染后的soup对象
    soup=BeautifulSoup(r.text,"html.parser")

    # 声明一个空列表以存储字典信息
    st=[{'股票编号':num},{'股票名称':name}]
    st1=msg("line1")
    st2=msg("line2")

    for i in range(11):
        st.append(st1[i])
        st.append(st2[i])
    stockmsg.append(st)

print(stockmsg)


 # 调用xlwt库，打开将写的表并添加sheet，同时标明列名
 # 读取源sql表
data_table = xlrd.open_workbook('../Desktop/stock_details.xls', formatting_info=True)
 # 复制源表
new_table = copy(data_table)
 # 获取所有sheet
sheeT1 = new_table.get_sheet('沪A')

for k in range(24):
    sheeT1.write(0,k,indexlist[k])
for v in range(1,sheet1.nrows):
    ff=stockmsg[v-1]
    for f in range(24):
        sheeT1.write(v,f,ff[f][indexlist[f]])
 # 保存为原地址，即完成修改
new_table.save('../Desktop/stock_details.xls')
