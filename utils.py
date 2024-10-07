import torch
import numpy as np
import matplotlib.pyplot as plt

def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0]
    return derivative

def burgers_equation(u, tx):
    u_tx = fwd_gradients(u, tx)
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    u_xx = fwd_gradients(u_x, tx)[:, 1:2]
    e = u_t + u*u_x - (0.01/np.pi)*u_xx
    return e

def ac_equation(u, tx):
    u_tx = fwd_gradients(u, tx)
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    u_xx = fwd_gradients(u_x, tx)[:, 1:2]
    e = u_t -0.0001*u_xx + 5*u**3 - 5*u
    return e 

def hj_equation(u, tx):
    u_tx = fwd_gradients(u, tx)
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    # e= u_t + 0.5* np.linalg.norm(u_x)**2
    e = u_t + 1/2*(u_x)**2
    return e

def bs_entropy_hj_equation(u, tx):
    u_tx = fwd_gradients(u, tx)
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    t=tx[:,0:1]
    x=tx[:,1:2]
    e= u_t + 1/t**2 *((x-t*x*u_x)*torch.log(1-t*u_x)+t*x*u_x)
    
    # u_xx = fwd_gradients(u_x, tx)[:, 1:2]
    # nu=1/10000
    # e=e+nu*u_xx
    
    
    # for i in range(len(e)):
    #     val=e[i]
    #     if torch.isnan(val):
    #         # e[i]=0 
    #         alter_loss=(1-t[i]/x[i]**2*u_x[i])**2
            
    #         if torch.isnan(alter_loss) or torch.isinf(alter_loss):
    #             e[i]=0
    #         else:
    #             e[i]=alter_loss
    #         # print('x-value'+str(x[i]),'t-value'+str(t[i]),'u_t-value'+str(u_t[i]),'u_x-value'+str(u_x[i]))
    return e 

def resplot(x, t, t_data, x_data, Exact, u_pred):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x, Exact[:,0],'-')
    plt.plot(x, u_pred[:,0],'--')
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.legend(['Reference', 'Prediction'])
    plt.title("Initial condition ($t=0$)")
    
    plt.subplot(2, 2, 2)
    t_step = int(0.25*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.legend(['Reference', 'Prediction'])
    plt.title("$t=0.25$")
    
    plt.subplot(2, 2, 3)
    t_step = int(0.5*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.legend(['Reference', 'Prediction'])
    plt.title("$t=0.5$")
    
    plt.subplot(2, 2, 4)
    t_step = int(0.99*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    plt.legend(['Reference', 'Prediction'])
    plt.title("$t=0.99$")
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.show()
    plt.close()
