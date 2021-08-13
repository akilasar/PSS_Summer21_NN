import numpy as np

# Global data used in the right hand side of the differential equations
mu      = 4.902793384008800e+03; # Gravity constant
a       = np.zeros([3,1]); # Engine acceleration

# Main simulation function
def main():

    # This sets up the initial state [r;v]
    h0      = 15;
    rM      = 1738;
    x       = np.zeros([6,1]);
    x[0]    =  -3.690075632373796e+02;
    x[1]    =   1.588986197430565e+03;
    x[2]    =   6.418452170490643e+02;
    x[3]    =  -2.783955398454346e-02;
    x[4]    =  -6.324843740303800e-01;
    x[4]    =   1.549806570220643e+00;
 
    dT      = 3.587142739135186e+00;
    tEnd    = 1000;
    n       = 100;
    
    # Read in the acceleration array
    aIn     = np.zeros([3,n])
 
    # Simulation loop
    for k in range(0,n):
        # Get the acceleration vector
        a = aIn[:,k]
        # Propagate one step
        x = RK4(x,dT)
        print(x)

# Numerical integration function using 4th order Runge Kutta
def RK4(x,h):

    ho2 = h/2
    k1  = RHS( x )
    k2  = RHS( x + ho2*k1 )
    k3  = RHS( x + ho2*k2 )
    k4  = RHS( x + h*k3 )

    x   = x + h*(k1 + 2*(k2+k3) + k4)/6
    
    return x

# Dynamical model
def RHS(x):

    r = np.zeros((3,1))
    v = np.zeros((3,1))
    r = x[0:3]
    v = x[3:6]

    magR = np.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])

    vDot = -mu*r/magR**3 + a

    xDot = np.vstack((v,vDot))

    return xDot
    
# Call main()
if __name__ == '__main__':
    main()
