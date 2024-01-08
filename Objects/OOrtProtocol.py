class OOrtProtocol:
    ask_for_metrics = "ask_for_metrics"
    ask_for_params = "ask_for_params"
    aggregation_from_server_to_client = "aggregation_s"
    aggregation_from_client_to_server = "aggregation_c"
    pickle_from_server_to_client = "pickle_s"
    pickle_from_client_to_server = "pickle_c"
    param_from_server_to_client = "param_s"
    param_from_client_to_server = "param_c"
    job_submit = "job"
    register_client = "register_c"
    ask_for_client_config = "register_xml_s"
    ask_for_job_config = "job_xml_s"
    shutdown = "shutdown"
    default = "None"
    end_symbol_s = "\x00\x00"
    end_symbol_b = b"\x00\x00"

    def __init__(self, command, jobName, modelMame, pickleSize, param_size, xmlSize, train_time, loss_2_sum):
        self.command = command
        self.jobName = jobName
        self.modelMame = modelMame
        self.pickleSize = pickleSize
        self.param_size = param_size
        self.xmlSize = xmlSize
        self.train_time = train_time
        self.loss_2_sum = loss_2_sum

    @staticmethod
    def newInstance(
            command, jobName, modelMame, pickleSize, param_size, xmlSize, train_time=0, loss_2_sum=0
    ):
        """
        构造一个通信包
        :param xmlSize: 发送xml文件的大小
        :param loss_2_sum:  损失平方和
        :param train_time:  训练时间
        :param command: 命令
        :param jobName: 任务的名字
        :param modelMame:  模型的名字
        :param pickleSize:  模型文件的大小
        :param param_size:  模型参数文件的大小
        :return:
        """
        return OOrtProtocol(command, jobName, modelMame, pickleSize, param_size, xmlSize, train_time, loss_2_sum)

    @staticmethod
    def newInstanceByContext(context):
        """
        通过字符串构造
        :param context:
        :return:
        """
        if context.endswith(OOrtProtocol.end_symbol_s):
            context = context.replace(OOrtProtocol, "")
        valList = context.split(" ")
        valList[3] = int(valList[3])
        valList[4] = int(valList[4])
        valList[5] = int(valList[5])
        valList[6] = int(valList[6])
        valList[7] = int(valList[7])
        return OOrtProtocol(*valList)

    def toString(self):
        """
        将协议内容转换为待发送的字符串
        :return:
        """
        return " ".join([self.command, self.jobName,
                         self.modelMame, str(self.pickleSize),
                         str(self.param_size), str(self.xmlSize),
                         str(self.train_time), str(self.loss_2_sum)])

# if __name__ == '__main__':
#     content = OOrtProtocol.newInstance(
#         command=OOrtProtocol.param_from_server_to_client,
#         jobName="jobName",
#         modelMame="modelName",
#         pickleSize="modelSize",
#         param_size="paramSize"
#     ).toString()
#     print(content)
