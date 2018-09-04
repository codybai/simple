# Sugar(nanny)说明



## 相关文档
使用请参考:[Sugar(nanny)使用手册](http://cf.meitu.com/confluence/pages/viewpage.action?pageId=50881155 "Sugar(nanny)使用手册")

后续更新:[Sugar(nanny)更新文档](http://cf.meitu.com/confluence/pages/viewpage.action?pageId=50881162 "Sugar(nanny)更新文档")

有功能需求/bug请上:[Sugar(nanny)家政后援团](http://cf.meitu.com/confluence/pages/viewpage.action?pageId=54308102 "Sugar(nanny)家政后援团")

## 分机测试步骤：
```
1.git clone下来
2.dataset  里的  train  下的tar包解压，使图片路径为  ***/Sugar/Dataset/train/*****.JPEG
3.修改train_AE_master.py(104行)  'master_addr':' 10.244.26.12',   地址
4.修改train_AE_member.py(101行)  'master_addr':' 10.244.26.12',   地址
5.用pycharm或者sh 分机执行  train_AE_master.py于  train_AE_member.py
```