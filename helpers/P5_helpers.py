import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import scipy.stats

def disp_img(ax, img, title=None, rand_color=False):
    if img.max() == 1 and img.dtype == np.uint8:
        img = img*255
    if rand_color:
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
        colors = np.random.randint(0, 255, size=(256, 1, 3)).astype(np.uint8)
        colors[0] = 255
        img_col = cv2.LUT(img, colors)
        ax.imshow(img_col)
    else:
        ax.imshow(img, cmap='gray')
    if title:
        ax.title(title)
    ax.axis(False)

def compute_contours(img):
    contours, cntr_img = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res_img = cv2.cvtColor(255*img, cv2.COLOR_GRAY2RGB)
    res_img = cv2.drawContours(res_img, contours, contourIdx=-1, color=(255, 0, 0), thickness=20)  # colorIdx==-1 : all contours
    return contours, res_img

def features(contours):
    feature_list = []
    for k, c in enumerate(contours):
        # major_axis/minor_axis are the square roots of the eigenvalues of the covariance matrix
        if len(c) > 5:
            (x,y), (minor_axis, major_axis), angle = cv2.fitEllipse(c)
        else:
            (x,y), (minor_axis, major_axis), angle = (0,0), (0,0), 0
        feature_list.append({
            'perimeter':  cv2.arcLength(c, True),
            'area':       cv2.contourArea(c),
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'ecc':        np.sqrt((major_axis-minor_axis)/major_axis) if major_axis>1e-12 else 0.,
        })
    return feature_list

def plot_feats(ax, feats, mu, sig, class_no):
    col = 'r' if class_no==0 else 'g'
    ax.plot(feats[:, 0], feats[:, 1], col+'s')
    ax.plot(mu[0], mu[1], col+'*', markersize=15)
    sx, sy = np.sqrt(sig[0][0]), np.sqrt(sig[1][1])
    ax.plot([mu[0]-sx, mu[0]+sx], [mu[1], mu[1]], col+':')
    ax.plot([mu[0], mu[0]], [mu[1]-sy, mu[1]+sy], col+':')
    ax.grid(True)

def visualize_model(feats_gb, mu_gb, sig_gb, feats_lic, mu_lic, sig_lic):
    x = np.linspace(0, 0.6, 100)
    y = np.linspace(0.6, 1, 100)
    X, Y = np.meshgrid(x,y)
    XY = np.dstack((X, Y))
    pdf_gb = scipy.stats.multivariate_normal(mean = mu_gb, cov = sig_gb).pdf
    pdf_lic = scipy.stats.multivariate_normal(mean = mu_lic, cov = sig_lic).pdf
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_feats(ax, feats_gb, mu_gb, sig_gb, 0)
    plot_feats(ax, feats_lic, mu_lic, sig_lic, 1)
    ax.contour(X, Y, pdf_gb(XY), cmap='Reds')
    ax.contour(X, Y, pdf_lic(XY), cmap='Greens')
    ax.set_title('Learned model')
    plt.show()
    
    print('Rendered in 3D (this may fail if you do not have installed the 3D support):')
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, pdf_gb(XY)+pdf_lic(XY), color='k')
    ax.set_title("Both pdfs")
    plt.show() 
    