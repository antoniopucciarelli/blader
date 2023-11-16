import matplotlib.pyplot as     plt 
from   src.geometryLIB   import camberline

def main():
    # variables allocation/definition
    stagger     = 30
    metalInlet  = -10
    metalOutlet = +70
    chord       = 10.0
    chebyschev  = True
    origin      = True
    number      = 2 
    pitch       = 10.0
    vector      = True 
    normalized  = False

    # object generation
    cLine = camberline.Camberline(stagger=stagger, metalIn=metalInlet, metalOut=metalOutlet, chord=chord, chebyschev=chebyschev, origin=origin)

    # axes generation
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plotting data
    cLine.plot(ax, normalized=normalized, number=number, pitch=pitch, vector=vector)

    # showing plot
    ax.set_aspect('equal')
    ax.grid(visible=True, linestyle='dotted')
    plt.show()

if __name__ == '__main__':
    main()