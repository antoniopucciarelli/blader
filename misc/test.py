import numpy as np 
import matplotlib.pyplot as plt 

# setting up camberline parameters
def nFunc(stagger, metalIn, metalOut) -> None:
    '''
    This function computes the `n` value used in the **Kulfan parametrization** :cite:p:`kulfan2008universal`. It uses the `camberline` object properties.  
    ''' 

    n = (np.tan(metalOut) + np.tan(metalIn)) / np.tan(stagger)

    print('>>> n = {0:+.3E}'.format(n))

    return n

def aFunc(metalOut, n) -> None:
    '''
    This function computes the ``a`` value used in the **Kulfan parametrization** :cite:p:`kulfan2008universal`. It uses the ``camberline`` object properties.  
    ''' 
    
    if n == 0: 
        a = 0
    else:
        a = np.tan(metalOut) / n 

    print('>>> a = {0:+.3E}'.format(a))

    return a

def bFunc(metalIn, n) -> None:
    '''
    This function computes the ``n`` value used in the **Kulfan parametrization** :cite:p:`kulfan2008universal`. It uses the ``camberline`` object properties.  
    ''' 
    
    if n == 0: 
        b = 0
    else:
        b = - np.tan(metalIn) / n

    print('>>> b = {0:+.3E}'.format(b))

    return b

def yFunc(a, b, n):

    x = np.linspace(0, 1, 100)

    y = a * x**n + b * (1 - x)**n

    print('>>> y = {0}'.format(y))

    return y

def main():

    stagger = 0
    metalIn = 50
    metalOut = -65

    stagger  = np.deg2rad(stagger)
    metalIn  = np.deg2rad(metalIn)
    metalOut = np.deg2rad(metalOut)

    n = nFunc(stagger=stagger, metalIn=metalIn, metalOut=metalOut)
    a = aFunc(metalOut=metalOut, n=n)
    b = bFunc(metalIn=metalIn, n=n)

    y = yFunc(a=a, b=b, n=n)

    plt.plot(np.linspace(0, 1, y.shape[0]), y, 'k')
    plt.show()

    return n, a, b

if __name__ == '__main__':
    main()
