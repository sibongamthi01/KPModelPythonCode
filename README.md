# KPModelPythonCode
This is the Python Code which was used by Mika Naidoo for her Masters thesis and it applies the KP model onto the given flux density data to produce predicted fluxes and condcut a grid search which finds the spectral age and magnetic field values which pproduce the minimum chi-square.








#GridSearchfortheNorthernLobeRegionofHydraA
import numpy as np
import math as m
from matplotlib import pyplot as plt



def fgrande(t):
    res=1.78*(t**0.3)*m.exp(-t)
    return res

def integral(x, y):
    n=x.size
    res=0.0
    for i in range(0, n-1):
        a=(x[i+1]-x[i])
        b=(y[i]+y[i+1])/2.
        res=res+a*b
    return res


def pachol_radio_data(k0, s, t, bmu, xdata):

#pro pachol_radio_data,k0,s,t,bmu,xdata,emiss
#;k0: normalizzazione iniziale
#;s: indice spettrale iniziale E^(-s)
#;t: tempo finale in Myr
#;bmu: campo magnetico in microG
#;xdata: asse delle frequenze in MHz
    tfin=t*1e6*3.156e7	#; in secondi
    gelm=1000
    assegel=np.zeros(gelm+1)
    for i in range(0, gelm):
        assegel[i]=10**(1.0+5.0*i/gelm)
    ngel0=np.zeros(gelm+1)
    for i in range(0, gelm):
        ngel0[i]=(assegel[i])**(-s)

    ngel=np.zeros(gelm+1)
    thm=1000

    asseth=np.zeros(thm+1)

    for j in range(0, thm+1):
        #asseth[j]=m.pi/18.0+(m.pi/2.0-m.pi/18.0)*j/thm
        asseth[j]=0.+(m.pi/2.)*j/thm

    z=0.055     #redshift
    intth=np.zeros(thm+1)

    assenu=xdata
    num=assenu.size


    neterm=1e-6#	;cm^-3
    r0=2.82e-13#		;cm
    me=0.511e-3#		;GeV
    c=3.0e10#			;cm/s
    nu0=2.8*bmu*1e-6#	;MHz
    nup=8980.*neterm**0.5*1e-6#		;MHz
    a=2.0*m.pi*m.sqrt(3.)*r0*me/c*1e6*nu0
    integel=np.zeros(gelm+1)
    emiss=np.zeros(num)

    for k in range(0, num):
        nu=assenu[k]
        for i in range(0, gelm):
            gel=assegel[i]
            x=2.*(nu/(3.*nu0*gel**2.))#*(1.+(gel*nup/nu)**2.)**(3./2.)
            for j in range(0, thm):
                theta=asseth[j]
                if (theta==0):
                    intth1=0.0
                else:
                    intth1=m.sin(theta)*fgrande(x/m.sin(theta))*m.sin(theta)*0.5
                bic=(1.37e-20)*(1.0+z)**4 # (1.0e-6) to delete
                bsyn=(1.94e-21)*(bmu*m.sin(theta))**2
                btot=bsyn+bic
                gelmax=1./(btot*tfin)
                if (gel<gelmax):
                    intth2=k0*(gel**(-s))*(1.0-btot*tfin*gel)**(s-2.0)
                else:
                    intth2=0.
                intth[j]=intth1*intth2
            integel[i]=a*integra(asseth,intth)
        emiss[k]=integra(assegel,integel)
    return emiss


def fit_norm(xdata,ydata,yerr,frad, val):
    nm=xdata.size
    fitfrad=np.zeros(nm)
    tempchi=np.zeros(nm)
    im=1000
    assenorm=np.zeros(im+1)
    for i in range(0, im+1):
        assenorm[i]=0.98+0.04*i/im
    chiq=np.zeros(im+1)
    for i in range(0, im+1):
        norm=assenorm[i]
        for n in range(0, nm):
            fitfrad[n]=norm*frad[n]
            tempchi[n]=((ydata[n]-fitfrad[n])**2)/(yerr[n])**2
        chiq[i]=np.sum(tempchi)
    chimin=np.min(chiq)
    normchimin=assenorm[np.argmin(chiq)]
    return chimin if val>0.0 else normchimin



def superfit_hydra(s, xdata, ydata, yerr):
    jm= 100
    bmin= 1
    bmax= 50
    asseb=np.zeros(jm)
    for j in range(0, jm):
        asseb[j]=bmin+(bmax-bmin)*j/(jm-1)
    km= 140
    tmin= 5
    tmax= 180
    asset=np.zeros(km)
    for k in range(0, km):
        asset[k]=tmin+(tmax-tmin)*k/km
    chi=np.zeros((jm, km))

    for j in range(0, jm):
        b=asseb[j]
        for k in range(0, km):
            t=asset[k]
            emiss=pachol_radio_data(s, t, b, xdata)
            chi[j,k]=fit_norm(xdata, ydata, yerr, emiss*1.821/emiss[4], 1.0)
            #print(100.0*j+k)
    chimin = np.min(chi)
    for j in range(0, jm):
        b=asseb[j]
        for k in range(0, km):
            t=asset[k]
            if (chi[j,k] == chimin):
                jbar=j
                kbar=k
                bbest=b
                tbest=t
                emiss=pachol_radio_data(s, tbest, bbest, xdata) # DP
                #print(emiss)
                norm2=fit_norm(xdata, ydata, yerr, emiss*1.821/emiss[4], -1.0)    # DP
                print(np.multiply(emiss, norm2*1.821/emiss[4])) # DP
    print(bbest,tbest,chimin)


    blow=0.
    bup=0.
    tlow=0.
    tup=0.

    if (jm >= 3):
        for j in range(0, jm-1):
            if ((chi[j,kbar] >= chimin+1) and (chi[j+1,kbar]<=chimin+1)):
                blow=abs(asseb[j]-bbest)
                print(j, blow, asseb[j])
            if ((chi[j,kbar] <= chimin+1) and (chi[j+1,kbar]>=chimin+1)):
                bup=abs(asseb[j+1]-bbest)
                print(j+1, bup, asseb[j+1])
    #print(blow,bup)

    if (km >= 3):
        for k in range(0, km-1):
            if ((chi[jbar,k] >= chimin+1) and (chi[jbar,k+1]<=chimin+1)):
                tlow=abs(asset[k]-tbest)
                print(k, tlow, asset[k])
            if ((chi[jbar,k] <= chimin+1) and (chi[jbar,k+1]>=chimin+1)):
                tup=abs(asset[k+1]-tbest)
                print(k+1, tup, asset[k+1])
    #print(tlow,tup)

    #return chimin, bbest, tbest


xdata = np.array([74.0, 330.0, 1000, 1100, 1330, 1485])
ydata = np.array([91.444, 16.802, 3.181, 2.682, 1.821, 1.494])
yerr = np.array([9.14, 1.68, 0.018672, 0.018989, 0.009504, 0.010558])



#lnxdata = np.log10(xdata)
#lnydata = np.log10(ydata)
#lnyerr = (1/np.log(10)) * np.float32(yerr)/np.float32(ydata)


superfit_hydra(2.00, xdata, ydata, yerr)
