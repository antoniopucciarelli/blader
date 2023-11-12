import matplotlib.pyplot     as     plt 
from   src.geometryLIB.blade import Blade

def main():
    # variables allocation/definition
    stagger     = 30
    metalInlet  = -40
    metalOutlet = +70
    chord       = 1.0
    chebyschev  = True
    origin      = True
    number      = 2
    pitch       = 3.0
    vector      = False
    normalized  = True

    # profile line 
    A          = [0.2, 0.3, 0.15]
    LEradius   = 3e-2 / 2
    TEradius   = 2.5e-2 / 2 
    wedgeAngle = 30

    # object generation
    # blade 
    blade = Blade(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chord=chord, pitch=pitch, Asuct=A, Apress=A, TEradius=TEradius, LEradius=LEradius, wedgeAngle=wedgeAngle, chebyschev=chebyschev, origin=origin)
    
    # axes generation
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plotting data
    # blade.plot(ax, normalized=normalized, number=number)

    # blade.update(stagger=40)

    blade.plot(ax, normalized=normalized, number=number)

    blade.save('blade.txt')

    # showing plot
    ax.set_aspect('equal')
    ax.grid(visible=True, linestyle='dotted')
    plt.show()

    blade.printout()

if __name__ == '__main__':
    main()