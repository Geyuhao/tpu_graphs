file_to_write = "./cleanedLog.log"
wf = open(file_to_write, "w")

with open('log.log') as f:
    lines = f.readlines()
    for line in lines:
        if "\x08" not in line and line != "\n":
            wf.writelines(line.strip()+"\n")
        
wf.close()
        
