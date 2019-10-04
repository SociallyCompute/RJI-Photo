class Record:

    file_pic_dict = {}
    file_rank_dict = {}

    def __init__(self, f):
        self.read_data(f)

    # given file path to read data from
    # expects the same format as we wrote
    def read_data(self, f):
        fi = open(f, 'r')
        for l in fi.readlines():
            li = l.split(';')
            self.file_pic_dict[li[0]] = li[1]
            self.file_pic_dict[li[0]] = li[2]

    # format is:
    # file_name;picture;ranking\n
    def write_data(self, f):
        fi = open(f, 'w')
        for i in range(len(self.file_pic_dict)):
            fi.write(self.file_pic_dict.keys()[i] + 
            ';' + self.file_pic_dict[self.file_pic_dict.keys()[i]] + 
            ';' + self.file_rank_dict[self.file_pic_dict.keys()[i]] + '\n')