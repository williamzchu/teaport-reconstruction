import numpy as np
import matplotlib.pyplot as plt
import camutils

def color_mask(imprefix, threshold):
   im1 = plt.imread(imprefix + "{:02d}".format(0) + '.png')
   im2 = plt.imread(imprefix + "{:02d}".format(1) + '.png')

   mask = np.sum(np.abs(im2 - im1), axis=-1) > threshold

   return mask

def reconstruct(colprefixL, colprefixR, c_threshold, imprefixL,imprefixR,threshold,camL,camR):
   colL = plt.imread(colprefixL + "{:02d}".format(1) + '.png')
   colR = plt.imread(colprefixR + "{:02d}".format(1) + '.png')
   
   #additional color mask for background foreground detection
   col_L_mask = color_mask(colprefixL, c_threshold)
   col_R_mask = color_mask(colprefixR, c_threshold)

   H_L, H_L_mask = camutils.decode(imprefixL,0,threshold)
   V_L, V_L_mask = camutils.decode(imprefixL,20,threshold)
   H_R, H_R_mask = camutils.decode(imprefixR,0,threshold)
   V_R, V_R_mask = camutils.decode(imprefixR,20,threshold)

   h, w = H_L.shape

   C_L = (H_L * H_L_mask + 1024*V_L * V_L_mask) * col_L_mask
   C_R = (H_R * H_R_mask + 1024*V_R * V_R_mask) * col_R_mask

   C_L = C_L.flatten()
   C_R = C_R.flatten()
   
   _, matchL, matchR = np.intersect1d(C_L, C_R, return_indices=True)

   xx,yy = np.meshgrid(range(w),range(h))
   xx = np.reshape(xx,(-1,1))
   yy = np.reshape(yy,(-1,1))
   pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
   pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)

   # Now triangulate the points
   pts3 = camutils.triangulate(pts2L, camL, pts2R, camR)

   #additionally setup correponding colors
   n = pts3.shape[1]
   col = np.empty((n,3))
   for i in range(n):
      col[i] = (colL[pts2L[1,i],pts2L[0,i]] + colR[pts2R[1,i],pts2R[0,i]]) /2
   
   return pts2L,pts2R,pts3,col

#test code
if __name__ == "__main__":
   threshold = 0.2
   colprefixL = "project/teapot/grab_1_u/color_C0_"
   colprefixL = "project/teapot/grab_1_u/color_C1_"
   colL = plt.imread(colprefixL + "{:02d}".format(1) + '.png')
   x,y,z = colL.shape
   print(colL)
   colL = colL.reshape((x*y,z), order = 'F')
   print(colL.shape)
   print(colL)