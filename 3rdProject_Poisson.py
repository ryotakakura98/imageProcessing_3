def poisson_blend(object_img, object_mask, bg_img, bg_ul):
    """
    Returns a Poisson blended image with masked object_img over the bg_img at position specified by bg_ul.
    Can be implemented to operate on a single channel or multiple channels
    :param object_img: the image containing the foreground object
    :param object_mask: the mask of the foreground object in object_img
    :param background_img: the background image
    :param bg_ul: position (row, col) in background image corresponding to (0,0) of object_img
    """
    #TO DO
    #object_img, penguin, back
    shape = object_img.shape
    constrains = ((shape[0]-2)+1)*(shape[1]-2) + ((shape[1]-2)+1)*(shape[0]-2)
    A = scipy.sparse.lil_matrix((constrains,(shape[0]-2)*(shape[1]-2)), dtype='double') # init lil
    C = np.zeros((constrains,1), dtype='double')
    penguin = object_img[1:(shape[0]-1),1:(shape[1]-1)]
    back = bg_img[bg_ul[0]:(bg_ul[0]+shape[0]),bg_ul[1]:(bg_ul[1]+shape[1])]

    #making the matrix
    row = constrains
    m = shape[0] - 2
    n = shape[1] - 2
    col = shape[0]*shape[1]
    topcount = 1
    rightcount = 0
    bottomcount = 1
    for i in range(0,row):
        if i <= m*3: #the left pixcels of penguin
            if i == 0: #left top of the penguin (with back)
                #penguin coordinate (0,0)
                A[0,0] = 1
                C[0,0] = back[0,1]+object_img[1,1]-object_img[0,1]
            elif i == (m*3-1): #left bottom of the penguin (with back)
                #penguin coordinate (0,m)
                A[i,m-1] = 1
                C[3*m-1,0] = back[m+1,1]+object_img[m,1]-object_img[m+1,1]
            elif (i-1)%3 == 0: #right side of the first column (with inside)
                #penguin coordinate ((i-1)//3,0)
                A[i,(i-1)//3] = 1
                A[i,(i-1)//3+m] = -1
                C[i,0] = penguin[(i-1)//3,0] - penguin[(i-1)//3,1]
            elif (i-2)%3 == 0: #down side of the first column (with back)
                #penguin coordinate ((i-2)//3,0)
                A[i,(i-2)//3] = 1
                A[i,(i-2)//3+1] = -1
                C[i,0] = penguin[(i-2)//3,0] - penguin[(i-2)//3+1,0]
            elif i%3 == 0: #left side of the first column (with back)
                #penguin coordinate ((i-3)//3,0)
                A[i,(i-3)//3] = 1
                C[i,0] = back[(i-3)//3+1,0]+object_img[(i-3)//3+1,1]-object_img[(i-3)//3+1,0]
        elif (i-m)%(m*2+1) == 0: #the top pixcels of penguin
            #penguin coordinate (0,topcount)
            A[i,topcount*m] = 1
            C[i,0] = back[0,topcount+1]+object_img[1,topcount+1]-object_img[0,topcount+1]
            topcount = topcount + 1
            pixcel_num = (topcount-1)*m
            count = 1 #count needs to be resetted at the top
        elif (i+1-m)%(m*2+1) == 0: #the bottom pixcels of penguin
            #penguin coordinate (m-1,bottomcount)
            A[i,(bottomcount+1)*m-1] = 1
            C[i,0] = back[m+1,bottomcount+2]+object_img[m,bottomcount+2]-object_img[m+1,bottomcount+2]
            bottomcount = bottomcount + 1
        elif i >= m+((2*m)*(n-1))+n: #the right pixcels of penguin
            #penguin coordinate (n-1,rightcount)
            if (i-(m+((2*m)*(n-1))+n))%2 == 0:
                A[i,(topcount-1)*m+rightcount] = 1
                C[i,0] = back[rightcount+1,n+1]+object_img[rightcount+1,n]-object_img[rightcount+1,n+1]
                if (i-(m+((2*m)*(n-1))+n))%2 > 0:
                    rightcount = rightcount + 1
            else:
                A[i,(topcount-1)*m+rightcount] = 1
                A[i,(topcount-1)*m+rightcount+1] = -1
                C[i,0] = penguin[rightcount-1,n-1] - penguin[rightcount,n-1]
        else:
            if count%2 == 0: #for the pixcels inside with with facing down
                A[i,pixcel_num] = 1
                A[i,pixcel_num+1] = -1
                C[i,0] = penguin[(count+1)//2-1,topcount-1]-penguin[(count+1)//2,topcount-1]
                pixcel_num = pixcel_num+1
                count = count + 1 #count is added as it gets down
            else: #for the pixcels inside with facing right
                A[i,pixcel_num] = 1
                A[i,pixcel_num+m] = -1
                C[i,0] = penguin[(count+1)//2-1,topcount-1]-penguin[(count+1)//2-1,topcount]
                count = count + 1 #count is added as it gets down



    #solve the lsqr
    v = scipy.sparse.linalg.lsqr(A.tocsr(), C); # solve w/ csr
    array = np.asarray(v[0])
    arrays = np.array_split(array, n)
    im_out = np.asarray(np.transpose(arrays))
    bg_img[bg_ul[0]+1:(bg_ul[0]+shape[0])-1,bg_ul[1]+1:(bg_ul[1]+shape[1])-1] = im_out[:,:]

    return bg_img
