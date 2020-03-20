#!/usr/bin/env python

import sys
import numpy as np
import scipy
import itertools

def calc_xydist(point,meanpoint):
    '''calculate the distance in x and y from the mean point'''
    xd = abs(point[0]-meanpoint[0])
    yd = abs(point[1]-meanpoint[1])
    return([xd,yd])

def read_pdb_get_Ps(pdbfile,chain1):
    '''get the xy coordinates of the two MSP proteins'''
    filename = pdbfile.split('/')[-1].split('.')[0]
    filedata = open(pdbfile,'r').readlines()
    
    cas = {}            #{chain{atomno:[x,y,z]}}
    for i in filedata:
        line= i.split()
        if len(i) > 25:
            if i[:4] == 'ATOM': 
                chain = i[21]
                atomtype = i[13:15]
                atomno = int(i[23:26])
                if atomtype == 'CA':
                    try:
                        cas[chain][atomno] =[i[31:38],i[38:47],i[47:55]]
                    except:
                        cas[chain] = {atomno:([i[31:38],i[38:47],i[47:55]])}
    ## get the 2 subunit' Ca coords
    L1Cas = []         # just the x and y coords of all subunit CAs
    for chain in cas:
        if chain == chain1:
            for aa in cas[chain]:
                L1Cas.append([float(x) for x in cas[chain][aa]])
    return(L1Cas)

def fit_plane(initial,XYZ,xscale,yscale):
    '''returns plane normal vector and two parallel vectors'''
    p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]
    
    def f_min(X,p):
        plane_xyz = p[0:3]
        distance = (plane_xyz*X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)
    
    def residuals(params, signal, X):
        return f_min(X, params)
    
    from scipy.optimize import leastsq
    sol = leastsq(residuals, p0, args=(None, XYZ))[0]
    #print("Solution: ", sol)
    #print("Old Error: ", (f_min(XYZ, p0)**2).sum())
    #print("New Error: ", (f_min(XYZ, sol)**2).sum())

    a,b,c,d = sol[0],sol[1],sol[2],sol[3]    
    #print('Equation for fit plane: {0}x + {1}y + {2}z + {3} = 0'.format(a, b, c, d))
    
    ## calculate plane normal:
    PNx = a
    PNy = b
    PNz = c
    

    #calculate plane vectors
    e1 = [c-b,a-c,b-a]
    e2 = [a*(b+c)-b**2-c**2,   b*(a+c)-a**2-c**2,   c*(a+b)-a**2-b**2]
    #print ('e1',e1)
    #print ('e2',e2)
    
    # get unit vector for plane vector and plane normal vector
    e1mag = np.sqrt((e1[0]**2)+(e1[1]**2)+(e1[2]**2))
    e1 = [(x/e1mag)*xscale for x in e1]

    e2mag = np.sqrt((e2[0]**2)+(e2[1]**2)+(e2[2]**2))
    e2 = [(x/e2mag)*yscale for x in e2]

    PN_mag = np.sqrt((PNx**2)+(PNy**2)+(PNz**2))
    PN_unit= [(x/PN_mag) for x in [a,b,c]]
    #print ('plane normal',PNx,PNy,PNz)
    #print ('plane normal unit vector',PN_unit)
    if PN_unit[2] < 0:
        PN_unit = [PN_unit[0],PN_unit[1],-PN_unit[2]]
        #print ('flipped plane normal unit vector',PN_unit)

    return(PN_unit,e1,e2,(a,b,c,d))

def plane_from_points(p1,p2,p3):
    '''returns ax+by+cy+d=0 from three points as np array'''
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    e1 = [c-b,a-c,b-a]
    e2 = [a*(b+c)-b**2-c**2,   b*(a+c)-a**2-c**2,   c*(a+b)-a**2-b**2]
    #print('Equation for starting plane: {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    return(a,b,c,-d)

def subunit_plane(Ca_coords,layername,color,chainname):
    '''fit a plane to the CA coordinates - starting with a plane perpendicular to the z axis as 1st guess'''
    bildout = open('{1}_{0}_plane.bild'.format(layername,chainname),'w')

    #### make starting plane    
    for ca in Ca_coords:
        bildout.write('.sphere {0} {1} {2} 0.5\n'.format(ca[0],ca[1],ca[2]))
    meanpoint = [np.mean([x[0] for x in Ca_coords]),np.mean([x[1] for x in Ca_coords]),np.mean([x[2] for x in Ca_coords])]
    startingplane = plane_from_points(np.array([meanpoint[0]+20,meanpoint[1]+20,meanpoint[2]]),np.array([meanpoint[0]-20,meanpoint[1]-20,meanpoint[2]]),np.array([meanpoint[0]-10,meanpoint[1]-15,meanpoint[2]]))
    #print('starting plane',startingplane)

    ### fit the actual coords using intial plabe as 1st guess
    xyz = np.array([[x[0] for x in Ca_coords],[x[1] for x in Ca_coords],[x[2] for x in Ca_coords]])
    
    ### calculate the proper scale
    allpoints_dists = []
    for point in Ca_coords:
        allpoints_dists.append(calc_xydist(point,meanpoint))
    maxx = max([x[0] for x in allpoints_dists])
    maxy = max([x[1] for x in allpoints_dists])

    ### fit the properly sized plane    
    PN,e1,e2,abcde = fit_plane(startingplane,xyz,maxx,maxy)
    
    ## calculate the fit plane
    tpp1 = [meanpoint[0]+e1[0],meanpoint[1]+e1[1],meanpoint[2]+e1[2]]
    tpp3 = [meanpoint[0]-e1[0],meanpoint[1]-e1[1],meanpoint[2]-e1[2]]
    tpp2 = [meanpoint[0]+e2[0],meanpoint[1]+e2[1],meanpoint[2]+e2[2]]
    tpp4 = [meanpoint[0]-e2[0],meanpoint[1]-e2[1],meanpoint[2]-e2[2]]

    ## draw the plane normal vector    
    bildout.write('.arrow {0} {1} {2} {3} {4} {5} 0.2 0.4 \n'.format(meanpoint[0],meanpoint[1],meanpoint[2],meanpoint[0]+(PN[0]*10),meanpoint[1]+(PN[1]*10),meanpoint[2]+(PN[2]*10)))
    bildout.write('.v {0} {1} {2} {3} {4} {5} \n'.format(meanpoint[0],meanpoint[1],meanpoint[2],meanpoint[0],meanpoint[1],meanpoint[2]+(10)))

    # draw the fit plane
    bildout.write('.color {0}\n'.format(color))
    bildout.write('.polygon {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'.format(tpp1[0],tpp1[1],tpp1[2],tpp2[0],tpp2[1],tpp2[2],tpp3[0],tpp3[1],tpp3[2],tpp4[0],tpp4[1],tpp4[2]))

    
    bildout.close()
    return(PN,meanpoint)

def angle_between_vecs(vector1,vector2):
    '''angle between the unit vector of two vectors in [i,j,k] format'''
    angle =np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))
    return(angle)


errmsg = '\nUSAGE: amyloid_angles.py <pdb file> <protofilament 1 chains> <protofilament2 chains>\n list chains like this ABCDE  GFHIJK\nmake sure they are in order'

try:
    thefile = sys.argv[1]
    chains1 = [x for x in sys.argv[2]]
    chains2 = [x for x in sys.argv[3]]
except:
    sys.exit(errmsg)

chains_n_planes1 = {}        #{chain:[PN vector,midpoint]} for protofil 1
chains_n_planes2 = {}        #{chain:[PN vector,midpoint]} for protofil 2
for i in chains1:
    chains_n_planes1[i] = subunit_plane(read_pdb_get_Ps(thefile,i),'subunit1','red',i)
for i in chains2:
    chains_n_planes2[i] = subunit_plane(read_pdb_get_Ps(thefile,i),'subunit2','blue',i)

print('intra-protofilament angles')
print('protofilament 1')

tmp_sums=[]
for i in itertools.combinations(chains1, 2):
    print (i,angle_between_vecs(chains_n_planes1[i[0]][0],chains_n_planes1[i[1]][0]))
    tmp_sums.append(angle_between_vecs(chains_n_planes1[i[0]][0],chains_n_planes1[i[1]][0]))
print('mean','SD',np.mean(tmp_sums),np.std(tmp_sums))

tmp_sums=[]
pf1angs = []
print('angles between plane normal vectors and z-axis')
for i in chains1:
    uv = chains_n_planes1[i[0]][0]
    ang = angle_between_vecs(uv,[0.0,0.0,1.0])
    print(i,ang)
    tmp_sums.append(ang)
    pf1angs.append(uv)
print('mean','SD',np.mean(tmp_sums),np.std(tmp_sums))

tmp_sums=[]
print('\nintra-protofilament angles')
print('protofilament 2')
for i in itertools.combinations(chains2, 2):
    ang = angle_between_vecs(chains_n_planes2[i[0]][0],chains_n_planes2[i[1]][0])
    print (i,ang)
    tmp_sums.append(angle_between_vecs(chains_n_planes2[i[0]][0],chains_n_planes2[i[1]][0]))
print('mean','SD',np.mean(tmp_sums),np.std(tmp_sums))

tmp_sums=[]
pf2angs = []
print('angles between plane normal vectors and z-axis')
for i in chains2:
    uv = chains_n_planes2[i[0]][0]
    ang = angle_between_vecs(uv,[0.0,0.0,1.0])
    print(i,ang)
    tmp_sums.append(ang)
    pf2angs.append(uv)
print('mean','SD',np.mean(tmp_sums),np.std(tmp_sums))


#### draw summary graphics
def draw_sum_bild(angs,name):
    sumbild = open('summary_{0}.bild'.format(name),'w')
    sumbild.write('.color red\n')
    for i in angs:
        ivals = [str(x) for x in i]
        sumbild.write('.v 0 0 0 {0}\n'.format(' '.join(ivals)))
    meanangi = np.mean([x[0] for x in angs])
    meanangj = np.mean([x[1] for x in angs])
    meanangk = np.mean([x[2] for x in angs])
    sumbild.write('.color yellow\n')
    sumbild.write('.v 0 0 0 {0} {1} {2} \n'.format(meanangi,meanangj,meanangk))
    sumbild.write('.color blue\n')
    sumbild.write('.v 0 0 0 0 0 1')
    sumbild.close()

draw_sum_bild(pf1angs,'protofil1')
draw_sum_bild(pf2angs,'protofil2')
