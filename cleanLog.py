import os

dirpath = '/home/yuhaoge2/TPU_Graphs_Cost_Model'
dirpaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.log') and 'Clean' not in f]

for dirpath in dirpaths:
    print(dirpath)
    file_to_write = dirpath.replace('.log', '_Clean.log')
    wf = open(file_to_write, "w")

    with open(dirpath) as f:
        lines = f.readlines()
        for line in lines:
            if "\x08" not in line and line != "\n" and "warnings" not in line and "UserWarning" not in line and "Generating Predictions:" not in line:
                wf.writelines(line.strip()+"\n")
                
    os.makedirs('log', exist_ok=True)
    os.rename(dirpath, dirpath.replace('train_', 'log/train_'))
            
    wf.close()
            
