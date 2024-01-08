from Connection.OOrtRequest import connectToSever
import argparse
from Objects.OOrtProtocol import OOrtProtocol
from Utils.XMLTool import parseXmlFileToDict
from Utils.FileUtil import get_file_size
from Client import sendFileToServer, sendStrContentToTarget, recvDataFromServer

parser = argparse.ArgumentParser(description="这个脚本用来向服务器发送一个工作配置文件，服务器将开启工作")
parser.add_argument("--job_path", type=str, help="工作定义的xml配置文件")
parser.add_argument("--mode", type=str, help="选用哪一种方法random/oort/YuanPi", default="YuanPi")

args = parser.parse_args()
xmlConfigFilePath = args.job_path
configDict = parseXmlFileToDict(xmlConfigFilePath)
print(configDict)
xmlSize = get_file_size(xmlConfigFilePath)
jobRequest = OOrtProtocol.newInstance(
    command=OOrtProtocol.job_submit,
    jobName=configDict["job"]["job_name"],
    modelMame="BlackNet",
    pickleSize=0,
    param_size=0,
    xmlSize=xmlSize
)
# 连接服务器
connectionSocket = connectToSever(configDict["target"]["ip"], int(configDict["target"]["port"]))
# 提交工作命令
sendStrContentToTarget(jobRequest.toString(), connectionSocket)
recvDataFromServer(connectionSocket)
# 发送XML文件
sendFileToServer(xmlConfigFilePath, connectionSocket)




