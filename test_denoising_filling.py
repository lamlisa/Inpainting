from project import *

################################# Load Images #################################

ocean = read_im("Img/ocean.jpg")
tree = read_im("Img/tree.jpg")
flowers = read_im("Img/flowers.jpg")

################################## Denoising ##################################

########## Parameters to modify ##########

im = flowers
show_im(im)

# ocean image
#line,col = 205,205 # horizon
#line,col = 100,395 # cloud
#line,col = 285,285 # ocean

# tree image
#line,col = 110,50 # horizon
#line,col = 100,148 # tree

# flowers image
#line,col = 105,215 # sun
line,col = 300, 200 # flowers

h = 21
noise_h = 50 # 200 for ocean, 50 for tree

##########################################

# original patch
patch_o = get_patch(line,col,h,im)
show_im(patch_o,"original patch")

# noise a part of the image
im_n2 = noise2(line,col,im,noise_h,0.5)
show_im(im_n2,"noised image")

# noised patch
patch_n = get_patch(line,col,h,im_n2)
show_im(patch_n,"noised patch")
vect_n = patch_to_vect(patch_n)

#uncomplete, complete = vect_dico(im,h) # dico from complete image
uncomplete,complete = vect_dico(im_n2,h) # dico from noised image

# test with different alpha values
for alpha in [0.01,0.001,0.0001,0.00001]:
    w, values, vect_d = denoise(vect_n,complete,alpha=alpha)
    print(len(np.where(w!=0)[0])) # nombre de composantes non nulles
    show_im(vect_to_patch(vect_d,h),"denoised patch alpha = {}".format(alpha))

################################### Filling ###################################

########## Parameters to modify ##########

im = tree
show_im(im,"original image")

# ocean image
#line,col = 205,205 # horizon -> width,height = 61,51
#line,col = 100,395 # cloud -> width,height = 61,51
#line,col = 285,285 # ocean -> width,height = 61,51

# tree image
#line,col = 110,50 # horizon -> width,height = 61,51
line,col = 100,148 # tree -> width,height = 31,31 ou 31,61

# flowers image
#line,col = 105,215 # sun -> 37,37
#line,col = 300, 200 # flowers -> 37,37

width = 31
height = 31

h = 11
alpha = 0.0001

########################################## 

# delete a rectangle
im_r = delete_rect(im,line,col,width,height)
show_im(im_r,"image with deleted rectangle")

# deleted rectangle
show_im(im[line-height//2:line+height//2+1,col-width//2:col+width//2+1],"deleted rectangle")

uncomplete, complete = vect_dico(im_r,h)
im_nr = filling(line,col,width,height,h,im_r,complete,alpha=alpha)
show_im(im_nr,"image after filling")