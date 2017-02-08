#
# morphing of two pictures 
#
# 

import sys
import cv2
import numpy as np

animals = { "tiger" : ('tiger.png', np.array([ np.array([108,132]), np.array([268,140]) ]) ),
			"raccoon":('raccoon.jpg',np.array([ np.array([430,345]),np.array([700,315]) ]) )
			}

def run(animal,human):
	print animal,human
	img1 = cv2.imread(animals.get(animal)[0])
	img2 = cv2.imread(human)
	
	#cv2.imshow('eyes',img1[142:178,290:326,:])
	#cv2.waitKey(0)

	print "shape of the image:", img1.shape

	# detect eyes and find affine transformation of one image into another
	eyes2 = get_eyes(img2)
	if eyes2 is None: 
		print "face or eyes are not detected in the image:",human
		return -1
	print eyes2
	if eyes2[0,0] > eyes2[1,0]: eyes2 = np.flipud(eyes2)

	eyes1 = animals.get(animal,None)[1]
	if eyes1 is None: 
		print animal,"is not available. Please choose one from:", animals.keys()
		return -1

	print "eyes detected:", eyes2
	# eyes centroids
	c1 = np.mean(eyes1, axis=0)
	c2 = np.mean(eyes2, axis=0)
	print "centroids:", c1,c2

	dx1,dy1 = eyes1[1]-eyes1[0]
	a1 = np.arctan2(dy1,dx1)

	dx2,dy2 = eyes2[1]-eyes2[0]
	a2 = np.arctan2(dy2,dx2)

	a = a2-a1
	print "angles:",a1,a2
	
	s = np.linalg.norm(eyes2[0]-eyes2[1])/np.linalg.norm(eyes1[0]-eyes1[1])
	#s = np.std(eyes2-c2)/np.std(eyes1-c1) - both variants are correct
	print "scale:",s
	

	# transform the second image and make the mask
	#M = transformation_from_points(eyes1, eyes2)
	dshape = img1.shape
	H,W,_ = dshape
	
	tr = np.array([c2 - (s*R(a).dot(c1))]).T
	print "translation:",tr

	M = np.hstack((s*R(a), tr)) 
	print "M:",M
	M = np.vstack([M, np.matrix([0.,0.,1.]) ])
	print "M:",M
	
	im22 = warp_im(img2,M,dshape)

	top = np.array([[195,156],[194,57],[202,0],[377,0],[377, 468]])
	bottom = np.array([[207,468],[271,423],[377,0],[377, 468]])
	center = np.array([[271,423],[235,290],[230,168]])

	points = [top, center, bottom]
	#corr_clr = correct_colours(img1, im22, 10.)
	mask = get_face_mask(img1,points)

	output_im = (img1 * (1.0 - mask) + im22 * mask).astype(np.uint8)

	cv2.imshow('mask',output_im)
	#cv2.imshow('human',im22)
	cv2.waitKey(0)

	# cut both images using their mask then combine them


def get_face_mask(im, landmarks):
	FEATHER_AMOUNT = 71
	im = np.zeros(im.shape[:2], dtype=np.float64)
	for points in landmarks:
		draw_convex_hull(im,
	                     points,
	                     color=1)

	im = np.array([im, im, im]).transpose((1, 2, 0))

	im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
	im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
	return im


def draw_convex_hull(im, points, color):
	cv2.convexHull(points)
	cv2.fillConvexPoly(im, points, color=color)


def get_eyes(img):
	face_cascade = cv2.CascadeClassifier('../../../opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('../../../opencv/data/haarcascades_cuda/haarcascade_eye.xml')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(faces) != 1: return None

	(x,y,w,h) = faces[0]
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	if len(eyes) != 2: return None

	(x1,y1,w1,h1),(x2,y2,w2,h2) = eyes
	return np.array([[x+x1+w1/2,y+y1+h1/2], [x+x2+w2/2,y+y2+h2/2]] )


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im



def correct_colours(im1, im2, dist):
	COLOUR_CORRECT_BLUR_FRAC = 0.6

	blur_amount = COLOUR_CORRECT_BLUR_FRAC * dist
	blur_amount = int(blur_amount)
	if blur_amount % 2 == 0:
		blur_amount += 1

	print "blurring area:",blur_amount
	im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
	im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

	# Avoid divide-by-zero errors.
	im2_blur += 128 * (im2_blur <= 1.0).astype(np.uint8)

	return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    print "R:",R

    print c2.T, (s2 / s1) * R.dot(c1.T)

    M = np.vstack([np.hstack(((s2 / s1) * R,
                                       (c2.T - (s2 / s1) * R.dot(c1.T)))),
                         np.array([0., 0., 1.])])
    
    return M


def R(a):
	return np.array([[np.cos(a), -np.sin(a)],
						 [np.sin(a), np.cos(a)]])                


if __name__ == "__main__":
	args = sys.argv
	if len(args) != 3:
		print "USAGE:\n  python morph.py <tiger|raccoon> <image>\n"
	else:
		animal,im = args[1:]
		run(animal,im)
