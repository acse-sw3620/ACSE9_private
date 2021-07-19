import matplotlib.pyplot as plt
import numpy as np



mlp = np.load('mlp.npy')
gan1 = np.load('GAN_no_noise.npy')
gan2 = np.load('GAN_with_noise.npy')

x = [i for i in range(mlp.shape[0])]    
plt.plot(x,mlp, label='mlp')
plt.plot(x,gan1, label='gan_no_loss')
plt.plot(x,gan2, label='gan_with_loss')
plt.title('MSE loss')
plt.legend()
plt.savefig(fname="comparation.png")
plt.show()


start = 500
plt.plot(x[start:],mlp[start:], label='mlp')
plt.plot(x[start:],gan1[start:], label='gan_no_loss')
plt.plot(x[start:],gan2[start:], label='gan_with_loss')
plt.title('MSE loss ('+str(start)+'-3000 epochs)')
plt.legend()
plt.savefig(fname='comparation_'+str(start)+'-3000.png')
plt.show()