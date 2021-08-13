from PIL import ImageOps, Image
sizes =[(300,215), (500, 357), (700, 500), (900, 643)]
size = [(300,215), (400, 285), (500, 357), (600, 430), (700, 500), (800, 571)]
for cw,ch  in sizes:
    im_a = Image.open('apollo113km.jpg')
    w = im_a.width 
    h = im_a.height
    print(w,h)
    print(cw,ch)
    wdif, hdif = (w-cw)//2, (h-ch)//2
    border = wdif, hdif, wdif, hdif  # left, top, right, bottom
    cropped_img = ImageOps.crop(im_a, border)
    cropped_img.save("cropped_"+str(cw)+'.png')
