paths = open("paths.txt", "rb")
for path in paths:
    with open(path, "rb") as f:
        img = f.read()
    img_string = str(img)
    xmp_start = img_string.find('<x:xmpmeta')
    xmp_end = img_string.find('</x:xmpmeta')
    if xmp_start != xmp_end:
        xmp_string = img_string[xmp_start:xmp_end+12]
        print(xmp_string)