import numpy as np
from scipy.stats import norm


def bs_entropy_hj_generator(num_t, num_x, typ='train'):
    N_f = num_t*num_x
    t = np.linspace(1e-3, 1, num_t).reshape(-1,1) # T x 1
    
    # Gaussian samples
    # x = np.linspace(1e-3, 1-1e-3, num_x)
    # mean = 0.5
    # std_dev = 0.1
    # x=norm.ppf(x, loc=mean, scale=std_dev).reshape(-1,1)
    # print(x)

    # # Uniform samples + More of 1/2
    x = np.linspace(1e-3, 1-1e-3, num_x * 9 // 10).reshape(-1,1) # N x 1
    x = np.concatenate((x, np.full((num_x - (num_x * 9 // 10), 1), 0.5)), axis=0)

    # # Uniform samples
    # x = np.linspace(1e-3, 1-1e-3, num_x).reshape(-1,1) # N x 1
    
    # # Extra Weight in middle 
    # num_samples=int(0.2*num_x)
    # x1= np.linspace(0, 0.39, num_samples)
    # x2= np.linspace(0.4, 0.6, num_x-2*num_samples)
    # x3= np.linspace(0.61, 1, num_samples)
    # x = np.concatenate((x1, x2, x3), axis=None).reshape(-1,1)
    
    
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T
    
    # Initial condition and boundary condition
    u = np.zeros((N, T))  # N x T
    u[:,0:1] = np.absolute(x-0.5)**2
    
    t_data = T_star.flatten()[:, None]
    x_data = X_star.flatten()[:, None]
    u_data = u.flatten()[:, None]
    
    t_data_f = t_data.copy()
    x_data_f = x_data.copy()
    
    if typ == 'train':
        idx = np.random.choice(np.where((x_data == x[0]) | (x_data == x[-1]))[0], num_t)
        t_data = t_data[idx]
        x_data = x_data[idx]
        u_data = u_data[idx]
        
        init_idx = np.random.choice(N-1, num_x-2, replace=False) + 1
        t_data = np.concatenate([t_data, np.zeros((num_x-2,1))], axis=0)
        x_data = np.concatenate([x_data, x[init_idx]], axis=0)
        u_data = np.concatenate([u_data, u[init_idx,0:1]], axis=0)
        
        return t_data, x_data, u_data, t_data_f, x_data_f
   
    else:
        return t_data_f, x_data_f
    
def hj_generator(num_t, num_x, typ='train'):
    N_f = num_t*num_x
    t = np.linspace(0, 1, num_t).reshape(-1,1) # T x 1
    x = np.linspace(-1, 1, num_x).reshape(-1,1) # N x 1
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T
    
    # Initial condition and boundary condition
    u = np.zeros((N, T))  # N x T
    # u[:,0:1] = np.power(x, 2)
    u[:,0:1] = np.absolute(x)
    
    t_data = T_star.flatten()[:, None]
    x_data = X_star.flatten()[:, None]
    u_data = u.flatten()[:, None]
    
    t_data_f = t_data.copy()
    x_data_f = x_data.copy()
    
    if typ == 'train':
        #boundary data at x=+-1
        idx = np.random.choice(np.where((x_data == -1) | (x_data == 1))[0], num_t)
        t_data = t_data[idx]
        x_data = x_data[idx]
        u_data = u_data[idx]
        
        #add in the initial data
        init_idx = np.random.choice(N-1, num_x-2, replace=False) + 1
        t_data = np.concatenate([t_data, np.zeros((num_x-2,1))], axis=0)
        x_data = np.concatenate([x_data, x[init_idx]], axis=0)
        u_data = np.concatenate([u_data, u[init_idx,0:1]], axis=0)
        # randomised t,x,u,data; u[init_idx,0:1] is exactly the initial values in random order
        
        return t_data, x_data, u_data, t_data_f, x_data_f
   
    else:
        return t_data_f, x_data_f
    
def bg_generator(num_t, num_x, typ='train'):
    N_f = num_t*num_x
    t = np.linspace(0, 1, num_t).reshape(-1,1) # T x 1
    x = np.linspace(-1, 1, num_x).reshape(-1,1) # N x 1
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T
    
    # Initial condition and boundary condition
    u = np.zeros((N, T))  # N x T
    u[:,0:1] = -np.sin(np.pi*x)
    
    t_data = T_star.flatten()[:, None] #change dimension to  N*T x 1
    x_data = X_star.flatten()[:, None] # N*T x 1
    u_data = u.flatten()[:, None]
    
    t_data_f = t_data.copy()
    x_data_f = x_data.copy()
    
    if typ == 'train':
        idx = np.random.choice(np.where((x_data == -1) | (x_data == 1))[0], num_t) 
        # np.where((x_data == -1) | (x_data == 1))[0] returns an array of all indices where x_data==-1 or 1, 
        t_data = t_data[idx]
        x_data = x_data[idx]
        u_data = u_data[idx]
        
        init_idx = np.random.choice(N-1, num_x-2, replace=False) + 1
        t_data = np.concatenate([t_data, np.zeros((num_x-2,1))], axis=0)
        x_data = np.concatenate([x_data, x[init_idx]], axis=0)
        u_data = np.concatenate([u_data, u[init_idx,0:1]], axis=0)
        
        return t_data, x_data, u_data, t_data_f, x_data_f
   
    else:
        return t_data_f, x_data_f

def ac_generator(num_t, num_x, typ='train'):
    N_f = num_t*num_x
    t = np.linspace(0, 1, num_t).reshape(-1,1) # T x 1
    x = np.linspace(-1, 1, num_x).reshape(-1,1) # N x 1
    T = t.shape[0]
    N = x.shape[0]
    T_star = np.tile(t, (1, N)).T  # N x T
    X_star = np.tile(x, (1, T))  # N x T
    
    # Initial condition and boundary condition
    u = np.zeros((N, T))  # N x T
    u[:,0:1] = (x**2)*np.cos(np.pi*x)
    # boundary condition part
    u[0,:] = -np.ones(T) 
    u[-1,:] = u[0,:]
    
    t_data = T_star.flatten()[:, None]
    x_data = X_star.flatten()[:, None]
    u_data = u.flatten()[:, None]
    
    t_data_f = t_data.copy()
    x_data_f = x_data.copy()
    
    if typ == 'train':
        idx = np.random.choice(np.where((x_data == -1) | (x_data == 1))[0], num_t)
        t_data = t_data[idx]
        x_data = x_data[idx]
        u_data = u_data[idx]
        
        init_idx = np.random.choice(N-1, num_x-4, replace=False) + 1
        t_data = np.concatenate([t_data, np.ones((2,1)), np.zeros((num_x-4,1))], axis=0)
        x_data = np.concatenate([x_data, np.array([[-1], [1]]), x[init_idx]], axis=0)
        u_data = np.concatenate([u_data, -np.ones((2,1)), u[init_idx,0:1]], axis=0)
        
        return t_data, x_data, u_data, t_data_f, x_data_f
   
    else:
        return t_data_f, x_data_f


num_t = 100
num_x = 256
t_data, x_data, u_data, t_data_f, x_data_f=bs_entropy_hj_generator(num_t, num_x)
# print(t_data)
# print(len(t_data))
print(len(x_data))
# print(len(u_data))
# print(t_data_f[:200])