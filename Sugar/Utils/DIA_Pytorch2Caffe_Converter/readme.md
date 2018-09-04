# pytorch2caffe（自动转换工具）

# Cartoon model transfer scripts

```
python main.py --sys_root examples/En_De_Code/compressed_model.py --model_name 2content_en1de3-morefat-gw5e4-ep72-nosharp_ud3-halfx2_s1t8_alllayer_gw2e5_ftgw1e5sharp4_2_net_G.pth --structure_model_name AE_ud3_halfx2

```
# Project Introduction
实现pytorch版 网络定义.py文件和.pth文件  到  .prototxt 和 .caffemodel 的转换 
    
目前测试过的模型(参考examples目录的内容)如下

- [x] lenet
- [x] vgg19
- [x] alexnet
- [x] resnet34
- [x] googlenet
- [x] squeezenet
- [x] enet

注意事项

1.由于pooling 操作在pytorch中out_width, out_height是向下取整， 而caffe中是向上取整。所以在 [caffeserver](http://mlabgit.meitu.com/DeepLearnUtilities/caffeserver) 新增MTPooling层来对应pytorch pooling层.

2.nn.BatchNorm2d 对应caffe中的BatchNorm+Scale的组合

3.nn.UpsamplinngNearest 对应caffe的Deconvolution层(kernel size和stride都等于scale_facetor)

4.不支持nn.Convtranspose2d 的output_padding参数为非0的情况。因为caffe没有对应的参数。


 
# 使用过程(这里以Lenet举例)

    1.把lenet.py(模型的类定义)和lenet.pth(模型的state_dict)放到同一个文件夹内（例如 examples/lenet/）。
    2.在lenet.py文件中额外定义一个名字叫 get_model_and_input的函数。函数返回模型instance和需要的Input Variable. 如下：
```python
    def get_model_and_input():
        pth_name = "lenet.pth"    
        pth_file = os.path.split(os.path.abspath(__file__))[0] +'/'+ pth_name
        print("pth file :", pth_file)
        model = LeNet()
        if os.path.isfile(pth_file):
            model.load_state_dict(torch.load(pth_file,map_location=lambda storage,loc: storage))
        else:
            print "Warning :Load pth_file failed !!!"

        batch_size = 1
        channels = 1
        height = 28
        width = 28
        images = Variable(torch.rand(batch_size,channels,height,width))
        return model, images    
```        
        
    3.执行 python main.py  examples/lenet/lenet.py  会在lenet.py所在的目录下生成相应的caffemodel和prototxt
  
