class DataSet:
    def __init__(self, *datas):
        self.datas = list(datas)

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, item):
        ret_list = []
        for data in self.datas:
            ret_list.append(data[item])
        return ret_list
