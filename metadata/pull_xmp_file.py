def find_color_code(data_loader):
    counter = 0
    i = 0
    for _,data,path in data_loader:
        i = i+1
        # path=path.rstrip()
        print(path)
        with open(path, "rb") as f:
            img = f.read()
        img_string = str(img)
        xmp_start = img_string.find('photomechanic:ColorClass')
        xmp_end = img_string.find('photomechanic:Tagged')
        if xmp_start != xmp_end:
            xmp_string = img_string[xmp_start:xmp_end]
            if xmp_string[26] != "0":
                print(xmp_string[26] + " " + path.decode('ascii') + "\n\n")
            else:
                counter = counter + 1
            # labels[path.decode('ascii')] = xmp_string[26]
        # print(label)
            if(xmp_string[26] == '1'):
                data[1] = torch.tensor(999)
            elif(xmp_string[26] == '2'):
                data[1] = torch.tensor(800)
            elif(xmp_string[26] == '3'):
                data[1] = torch.tensor(700)
            elif(xmp_string[26] == '4'):
                data[1] = torch.tensor(650)
            elif(xmp_string[26] == '5'):
                data[1] = torch.tensor(500)
            else:
                data[1] = torch.tensor(250)
    print(counter)
    print("Total Images: " + str(i))