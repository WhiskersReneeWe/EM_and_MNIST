
library(mnormt)
library(matrixcalc)
library(mvtnorm)
library(slam)
library(svMisc)

load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }

load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }

 print('loading training images ...')
 train_images <- load_image_file('train-images-idx3-ubyte')
 print('loading training labels')
 train_labels <- load_label_file('train-labels-idx1-ubyte')
 print('Loading job done!')


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

########### test check images and labels matching###############
# train_images$x refers to all the images, where each image is a row vector 
train_imgs = train_images$x
show_digit(train_imgs[1,]) 
# train_labels[1]

########### Done checking and commenting this out ############

# Shrink the image
# Divide each 28*28 image into 2*2 non-overlapping blocks. 
shrink_img <- function(img){
  # input: 28 * 28 image
  # window size is 2 * 2, move horizontally -- fix row first, move across columns
  # output: 14 * 14 image
  shrinked_img = matrix(, nrow = 14, ncol = 14)
  
  for (i in c(0:13)) {
      for (j in c(0:13)){
        row = 1 + 2 * i
        col = 1 + 2 * j
        small_block = img[row:row+1, col:col+1]
        ave = sum(small_block)/4
        # store it
        shrinked_img[i+1, j+1] = ave
      }
    
  }
  return(shrinked_img)
}


########### Test shrink_img() function ########
# uncomment it to test

# get one original image, wrap it to 28 * 28, plot it
an_img = matrix(train_imgs[100, ], nrow = 28)
show_digit(an_img)
#shrink it and plot
image(shrink_img(an_img))
################################################

# IMAGES - {0, 1, 2, 3, 4} and corresponding LABELS
temp = as.data.frame(train_imgs)
temp$labels = train_labels
train = temp[temp$labels < 5, ]
LABELS = train$labels
IMAGES = subset(train, select=-c(labels))
IMGS = as.matrix(IMAGES) 


# to plot an image from IMAGE
test_image = matrix(IMGS[25, ], nrow = 28)
show_digit(matrix(test_image, nrow = 28))
image(shrink_img(test_image))

########### test done #######

# shrink training images into n * 196
s_train_imgs = matrix(, nrow = nrow(IMGS), ncol = 196)
for (i in 1:nrow(IMGS)) {
  img = IMGS[i, ]
  img = matrix(img, nrow = 28)
  s_img = shrink_img(img)
  s_img = matrix(s_img, nrow = 1)
  s_train_imgs[i, ] = s_img
}
mode(s_train_imgs) <- "integer"

imgs = s_train_imgs / 255
# sample 1000 imgs from the orginial imgs

set.seed(111109)
sampled_rows = sample(1:nrow(imgs), 1000, replace = TRUE)
imgs = imgs[sampled_rows, ]

######## Initiation ############################################
init_theta <- function(){
# initialize mu_0, mu_1, ..., mu_4
MU = matrix(, nrow = 5, ncol = 196)
for (i in 1:5){
  MU[i, ] = matrix(sample(1:255, 196, replace=TRUE)/255)
}

# initialize 5 Covariance Matrix - Spherical Guassian
I_sph = diag(1, 196, 196)


cov_mats = list()
vars = c()
for (i in 1:5){
  vars = c(vars, runif(1)^2) + 0.05
  cov_mats[[i]] = vars[i] * I_sph
}

# initialize PIs
x = runif(5, 0, 1)
pi = x/sum(x)

init_list = list(MU, cov_mats, pi)

return (init_list)

}



Fij <- function(xi, piii, MUUU, COVVV){
  # J -- cluster index
  Fi_j = c()
  
  F_ij_denom = 0
  for (j in 1:5){
    F_ij_denom = F_ij_denom + piii[j] * dmvnorm(xi, mean = MUUU[j, ], sigma = COVVV[[j]])
  }
  
  for (j in 1:5){
    top_g = piii[j] * dmvnorm(xi, mean = MUUU[j, ], sigma = COVVV[[j]])
    Fi_j = c(Fi_j, top_g / F_ij_denom)
  }
  
  return(Fi_j)
}


# Test Fij - Fij(img[1, ]) should give 5 probabilities that sum to 1: 
#sum(Fij(imgs[2, ], p, MU, cov_mats)) == 1
#Passed test

inits = init_theta()
p = inits[[3]] #p(z)
MU = inits[[1]]
cov_mats = inits[[2]]
# initialize log-likelihood
prevLLK = -1e300
tol = 1e-8
######## E-step ?########

for (iter in 1:100){
  progress(iter, progress.bar = TRUE)
  Sys.sleep(0.01)
  if (iter == 100) message('Done!')
  
  F_MAT = matrix(, nrow = nrow(imgs), 5)
  
  for (i in 1:nrow(imgs)){
  F_MAT[i, ] = Fij(imgs[i, ], p, MU, cov_mats)
  }
  
  MU = matrix(, nrow = 5, ncol = 196)
  for (j in 1:5){
    MU[j, ] = colSums(imgs * F_MAT[ ,j])/ sum(F_MAT[ ,j])
  }
  
  # Alternative
  # MU = matrix(, nrow = 5, ncol = 196)
  # for (j in 1:5){
  #   Fij_sum_t = 0
  #   Fij_sum_b = 0
  #   
  #   for (i in 1:nrow(imgs)){
  #   
  #     Fij_sum_t = Fij_sum_t + Fij(imgs[i, ], pi = p, mu = MU, cov_var = cov_mats)[j] * imgs[i, ]
  #     Fij_sum_b = Fij_sum_b + Fij(imgs[i, ], pi = p, mu = MU, cov_var = cov_mats)[j]
  #   }
  #   
  #   MU[j, ] = Fij_sum_t / Fij_sum_b
  #   
  # }
  
  ############ update p
  p = colSums(F_MAT) / nrow(F_MAT)  # p(z)
  
  ########### updates cov-mats
  cov_mats = list()
  vars = c()
  for (j in 1:5) {
    mean_j = matrix(1, nrow = nrow(imgs)) %*% MU[j, ]
    dev_j = imgs - mean_j # deviation in group 1
    top_var_j = sum(F_MAT[ ,j] * row_norms(dev_j))
    var_j = top_var_j / (196 * sum(F_MAT[ ,j]))
    vars = c(vars, var_j) 
    vars_added = vars + 0.05
    cov_mats[[j]] = vars_added[j] * I_sph
  }
  

  
  # calculate log-likelihood
  ll = matrix( , nrow = nrow(imgs), ncol = 1)
  for (i in 1:nrow(imgs)){
    inner_sum = matrix( , nrow = 5, ncol =1)
    for (j in 1:5){
      inner_sum[j, ] = p[j] * dmvnorm(imgs[i, ], mean = MU[j, ], sigma = cov_mats[[j]])
    }
    ll[i, ] = log(sum(inner_sum))
  }

  
  currLLK = sum(ll)
  print(currLLK)
  prevLLK <- currLLK

}
