def plot_trajectory(datax,datay,perceptron,step=10):
    plt.figure()
    w_histo, f_histo, grad_histo = perceptron.fit(datax,datay)
    xmax, xmin = np.max(w_histo[:,0]), np.min(w_histo[:,0])
    ymax, ymin = np.max(w_histo[:,1]), np.min(w_histo[:,1])
    dev_x, dev_y = abs(xmax-xmin)/4, abs(ymax-ymin)/4 # defines a margin for border
    dev_x += int(dev_x == 0)*5 # avoid dev_x = 0
    dev_y += int(dev_y == 0)*5
    grid,x1list,x2list=make_grid(xmin=xmin-dev_x,xmax=xmax+dev_y,ymin=ymin-dev_y,ymax=ymax+dev_y)
    plt.contourf(x1list,x2list,np.array([perceptron.loss(datax,datay,w)\
                                         for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.scatter(w_histo[:,0], w_histo[:,1], marker='+', color='black')
    plt.show()