

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]  # 去掉注释
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces  去掉多余的空格
    module_defs = []  
    for line in lines:
        if line.startswith('['): # This marks the start of a new block  [ 代表的是一个块的开始
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()  # 块的类别 如 net、convolutional、shortcut、yolo、route、upsample 等
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:  # 每个块下面的内容如  batch_normalize、size 等参数，形成键值对
            key, value = line.split("=")
            value = value.strip()  # 去掉多余的空格
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs  # 返回是一个列表，列表中是每一个块的字典

def parse_data_config(path):
    """
    Parses the data configuration file
    解析数据配置文件、
   
    返回的是字典
    包括以下内容
    classes= 1  # 类别数
    train=data/custom/train.txt
    valid=data/custom/valid.txt
    names=data/custom/classes.names
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
