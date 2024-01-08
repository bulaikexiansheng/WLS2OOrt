import xml.etree.ElementTree as ET
from Utils.FileUtil import readFileToString


def element_to_dict(element):
    # print(element)
    """将XML元素转换为字典"""
    result = {}
    for child in element:
        if child:
            # 递归处理子元素
            child_data = element_to_dict(child)
        else:
            # 处理叶子节点
            child_data = child.text if child.text else None

        # 检查是否已经有相同的键
        if child.tag in result:
            # 如果已经有相同的键，则将其转换为列表
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_data)
            else:
                result[child.tag] = [result[child.tag], child_data]
        else:
            result[child.tag] = child_data
    return result


def parseXmlFileToDict(xmlFilePath):
    """
    将xml文件转换为字典形式的数据结构
    :param xmlFilePath:
    :return: xml文件的字典表示形式
    """
    xmlContent = readFileToString(xmlFilePath)
    return element_to_dict(ET.fromstring(xmlContent))


def parseXmlStringToDict(xmlString):
    """
    将xml字符串转换为字典形式的数据结构
    :param xmlString: xml字符串
    :return: xml文件的字典表示形式
    """
    return element_to_dict(ET.fromstring(xmlString))
