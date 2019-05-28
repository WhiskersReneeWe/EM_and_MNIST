########## 3000 DATA POINTS##########
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
#test <<- load_image_file('mnist/t10k-images-idx3-ubyte')
print('loading training labels')
train_labels <- load_label_file('train-labels-idx1-ubyte')
#test$y <<- load_label_file('mnist/t10k-labels-idx1-ubyte')  
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
set.seed(111109)

# get one original image, wrap it to 28 * 28, plot it
# an_img = matrix(train_imgs[100, ], nrow = 28)
# show_digit(an_img)
# #shrink it and plot
# image(shrink_img(an_img))
################################################

# IMAGES - {0, 1, 2, 3, 4} and corresponding LABELS
temp = as.data.frame(train_imgs)
temp$labels = train_labels
train = temp[temp$labels < 5, ]
sampled_rows = sample(1:nrow(train), 3000, replace = TRUE)
sample_train = train[sampled_rows, ]
LABELS = sample_train$labels
IMAGES = subset(sample_train, select=-c(labels))
IMGS = as.matrix(IMAGES) 


# to plot an image from IMAGE
test_image = matrix(IMGS[5, ], nrow = 28)
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
I_sph = diag(1, 196, 196)
# imgs contains 4000 imgs from the orginial imgs



######## Initiation ############################################
######## Initiation method 1-  All random ############################################
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


############# calculate Fij Matrix
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

########### Generate population MU, p, Cov_mats ################
MU_pop = matrix(, nrow = 5, ncol = 196)
for (i in 1:5){

  m = colSums(imgs[LABELS == (i-1), ])/nrow(imgs[LABELS == (i-1), ])
  MU_pop[i, ] = m

}

cov_mats_pop = list()
vars = matrix( , nrow = 5, ncol =196)
for (j in 1:5) {
  imgs_j = imgs[LABELS == (j-1), ]
  MU_j = MU_pop[j, ]
  var_j = c()
  for (d in 1:196){
    diff_tol = 0
    for (i in 1:nrow(imgs_j)){
      diff = imgs_j[i, ][d] - MU_j[d]
      diff_tol = diff_tol + diff**2

    }
    var_j =  c(var_j, diff_tol/nrow(imgs_j))
  }
  vars[j, ] = var_j + 0.05
  cov_mats_pop[[j]] = vars[j, ] * diag(1, 196, 196)

}

p_pop = c()
for (i in 1:5){
  label_i = i - 1
  p_i = sum(LABELS == label_i)/nrow(imgs)
  p_pop = c(p_pop, p_i)
}

init_gaussian_MU<- function(){
  # initialize mu_0, mu_1, ..., mu_4
  MU = matrix(, nrow = 5, ncol = 196)
  for (i in 1:5){
    #MU[i, ] = matrix(sample(1:255, 196, replace=TRUE)/255) # sample mean-var init method 1
    MU[i, ] = rmvnorm(n = 1, mean = MU_pop[i, ], sigma = cov_mats_pop[[i]])
  }
  
  # initialize 5 Covariance Matrix - Spherical Guassian
  I_sph = diag(1, 196, 196)
  
  return(MU)
  
}

# mv.dnorm <- function(x, mu, cov_mat) {   # function definition // HL
#   inv.cov_mat <- chol(solve(cov_mat)) # compute cholesky decomposition of inverse covariance // HL
#   det.cov <- det(cov_mat)              # compute determinant
#   diff <- matrix((x - mu), nrow = 196)       
#   dnorm = exp(-0.5 * t(diff) %*% inv.cov_mat %*% diff) / (det.cov^0.5 * (2*3.14)^(196/2))
#   return(dnorm)# evaluates likelihoods at once // HL
# }

############# Call Init Method 1 - all random case
# inits = init_theta()
# p = inits[[3]] #p(z)
# MU = inits[[1]]
# cov_mats = inits[[2]]
# ############# uncommnent this when needed ###############

############## Call Init Method 2: - used population mean, P, cov_mats
# p = p_pop
# MU = MU_pop
# cov_mats = cov_mats_pop
############## Uncomment Method 2 when needed ########################

############## Call Init Method 3: - Gaussian group means, population cov_mats, population p
p = p_pop
MU = init_gaussian_MU()
cov_mats = cov_mats_pop
############## Uncomment Method 3 when needed ########################


# initialize log-likelihood
prevLLK = -1e300
######## E-step ?########
for (iter in 1:30){
  progress(iter, progress.bar = TRUE)
  Sys.sleep(0.01)
  if (iter == 10) message('Done!')
  
  F_MAT = matrix(, nrow = nrow(imgs), 5)
  
  for (i in 1:nrow(imgs)){
    F_MAT[i, ] = Fij(imgs[i, ], p, MU, cov_mats)
  }
  
  # calculate log-likelihood
  ll = matrix( , nrow = nrow(imgs), ncol = 1)
  for (i in 1:nrow(imgs)){
    inner_sum = matrix( , nrow = 5, ncol =1)
    for (j in 1:5){
      inner_sum[j, ] = p[j] * dmvnorm(imgs[i, ], mean = MU[j, ], sigma =cov_mats[[j]])
    }
    ll[i, ] = log(sum(inner_sum))
  }
  
  
  currLLK = sum(ll)
  
  
  print(prevLLK)
  prevLLK <- currLLK
  
  
  MU = matrix(, nrow = 5, ncol = 196)
  for (j in 1:5){
    MU[j, ] = colSums(imgs * F_MAT[ ,j])/ sum(F_MAT[ ,j])
  }
  
  ############ update p
  p = colSums(F_MAT) / nrow(F_MAT)  # p(z)
  
  ########### update cov-mats
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
}
   
########### Calculate Classification Error ###############
FIJ_MAT = matrix(, nrow = nrow(imgs), 5)

for (i in 1:nrow(imgs)){
  FIJ_MAT[i, ] = Fij(imgs[i, ], p, MU, cov_mats)
}

classification = apply(FIJ_MAT, 1, which.max)
class = classification - 1
classified_lables = class

# classification accuracy 
sum(classified_lables == LABELS)/nrow(imgs)

# Estimated Ps 
# 0.11144670 0.10416519 0.05939797 0.07339108 0.65159905
# Real Ps
# 0.20025 0.22775 0.18975 0.19275 0.18950
# real_p = c()
# for (i in 1:5){
#   label_i = i - 1
#   p_i = sum(LABELS == label_i)/3000
#   real_p = c(real_p, p_i)
# }
# 
# real_p




