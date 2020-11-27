import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

########################### Inpainting functions ##############################

MISSING = -100
       
def read_im(fn):
    """Read image and normalize it"""
    return plt.imread(fn)/255

def show_im(data,title=""):
    """Display image"""
    plt.figure()  
    plt.imshow(data)
    plt.title(title)
    
def get_patch(i,j,h,im):
    """
    Get the patch centered on (i,j) from the image im
    (patch of size h if h is an odd number, of size h+1 otherwise)
    """
    n, m, _ = im.shape
    r = h//2
    # if patch out of range
    if i-r<0 or i+r>=n or j-r<0 or j+r>=m:
        print('Out of range')
        return None
    else:
        return im[i-r:i+r+1,j-r:j+r+1]

def patch_to_vect(patch):
    """Convert the patch into a column vector"""
    return patch.reshape(-1)

def vect_to_patch(vect,h):
    """Convert the colomn vector into a patch"""
    return vect.reshape(h,h,3)

def noise(im,prc):
    """Delete prc% of pixels randomly chosen"""
    n, m, _ = im.shape
    p = int(n*m*prc)
    imc = im.copy()
    range_i = np.random.choice(range(n),p)
    range_j = np.random.choice(range(m),p)
    imc[range_i,range_j] = MISSING
    return imc

def noise2(i,j,im,h,prc):
    """
    Return the image with prc% of noise in the patch centered in (i,j)
    """
    im2 = im.copy()
    p = int(h*h*prc)
    patch = get_patch(i,j,h,im2)
    range_i = np.random.choice(range(h),p)
    range_j = np.random.choice(range(h),p)
    patch[range_i,range_j] = MISSING
    return im2
  
def delete_rect(im,i,j,width,height):
    """
    Delete from the image a rectangle centered on (i,j) of size width*height
    (width+1 if even number, same for height)
    """
    n, m, _ = im.shape
    w = width//2
    h = height//2
    # if patch out of range
    if i-h<0 or i+h>=n or j-w<0 or j+w>=m:
        print('Out of range')
        return None
    else:
        imc = im.copy()
        imc[i-h:i+h+1,j-w:j+w+1] = MISSING
        return imc
    
def vect_dico(im,h):
    """
    Return 2 array : one containing the vectors with noise, one containing the 
    vectors without noise
    """    
    uncomplete = []
    complete = []
    n, m, _ = im.shape
    step = h
    i = step
    while(i<(n-h)):
        j = step
        while(j<(m-h)):
            patch = get_patch(i,j,h,im)
            if MISSING in patch:
                uncomplete.append(patch_to_vect(patch))
            else:
                complete.append(patch_to_vect(patch))
            j += step
        i += step
    return np.array(uncomplete), np.array(complete)

def denoise(vect,complete,alpha=0.001,max_iter=50000):
    """Denoise the noised vector, using the dictionary with complete patches""" 
    # use lasso to determine the weight for each complete patch which best 
    # approximates the non noise pixels
    lasso = Lasso(alpha=alpha,max_iter=max_iter)
    indices_train = np.where(vect != MISSING)[0]
    indices_test = np.where(vect == MISSING)[0]
    lasso.fit(complete.T[indices_train],vect[indices_train])
    
    # predict the values of the noised pixels
    values = lasso.predict(complete.T[indices_test])
    
    # replace the noise in the vector by the values
    vect2 = vect.copy()
    vect2[indices_test] = values
    return lasso.coef_, values, vect2

def approximate(vect,complete,alpha=0.001,max_iter=50000):
    """Approximate the noised vector, using the dictionary with complete patches"""
    # use lasso to determine the weight for each complete patch which best 
    # approximates the non noise pixels
    lasso = Lasso(alpha=alpha,max_iter=max_iter)
    indices_train = np.where(vect != MISSING)[0]
    lasso.fit(complete.T[indices_train],vect[indices_train])
    
    # predict the values of the noised pixels
    vect_predict = lasso.predict(complete.T)
    return lasso.coef_, vect_predict
           
def edge_pixels_rect(l,c,width,height):
    """
    Return the pixels of the edge of the patch where (l,c) is the top-left corner
    (clockwise order)
    """
    pixels = list(zip([l]*width, range(c,c+width+1)))
    pixels += list(zip(range(l+1,l+height), [c+width-1]*(height-1)))
    pixels += list(zip([l+height-1]*(width-1), range(c+width-2,c-1,-1)))
    pixels += list(zip(range(l+height-2,l,-1), [c]*(height-2)))
    return pixels

def update(i,j,im,h,complete,alpha=0.001,max_iter=50000):
    """Update the image by denoising the patch centered on (i,j)"""
    vect = patch_to_vect(get_patch(i,j,h,im))
    
    lasso = Lasso(alpha=alpha,max_iter=max_iter)
    indices_train = np.where(vect != MISSING)[0]
    indices_test = np.where(vect == MISSING)[0]
    lasso.fit(complete.T[indices_train],vect[indices_train])
    
    # predict the values of the noised pixels
    values = lasso.predict(complete.T[indices_test])
    
    # replace the noise in the vector by the values
    vect[indices_test] = values
    r = h//2
    
    # update the image
    im[i-r:i+r+1,j-r:j+r+1] = vect_to_patch(vect,h)
    
def filling(i,j,width,height,h,im,complete,alpha=0.001,max_iter=50000):
    """
    Fill the rectangle centered on (i,j) of size height*width with patch 
    centered on the first missing pixel met in a clockwise order
    """
    r = h//2 +1
    im2 = im.copy()
    l, c = i-height//2, j-width//2
    tmp_w, tmp_h = width, height
    missing_pixel = True
    while missing_pixel:
        for p,q in edge_pixels_rect(l,c,tmp_w,tmp_h):
            if (im2[p,q] == MISSING).all():
                update(p,q,im2,h,complete,alpha=alpha)
        l += r
        c += r
        tmp_w -= 2*r
        tmp_h -= 2*r
        if (im2[l,c] != MISSING).all():
            missing_pixel = False
    return im2
