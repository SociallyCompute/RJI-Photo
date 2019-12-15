def find_color_code(data_loader):
    counter = 0
    i = 0
    for _,_,path in data_loader:
        i = i+1
        path=path.rstrip()
        with open(path, "rb") as f:
            img = f.read()
        img_string = str(img)
        xmp_start = img_string.find('photomechanic:ColorClass')
        xmp_end = img_string.find('photomechanic:Tagged')
        if xmp_start != xmp_end:
            xmp_string = img_string[xmp_start:xmp_end]
            if xmp_string[26] != "0":
                print(xmp_string[26] + " " + str(path) + "\n\n")
            else:
                counter = counter + 1
    print(counter)
    print("Total Images: " + str(i))