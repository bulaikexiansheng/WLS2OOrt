# WLS2OOrt
## 项目运行方法
1. 修改目录中服务器配置文件server_config.xml中的ip地址为本机目前在使用的ip地址和需要监听的端口信息
2. 修改客户端注册文件client_register.xml中服务器的ip地址和端口信息
3. 修改工作提交者工作配置文件mnist.xml/fashion_mnist.xml/cifar_10.xml的目标服务器的ip地址和端口信息
4. 开启三个powershell界面
  - 界面1：运行`python Server.py`
  - 界面2：运行`bash runClient.sh`
  - 界面3：运行`bash job.sh`
5. 如果需要切换训练任务，可以`vim job.sh`       
   