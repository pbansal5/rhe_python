  #! /usr/bin/python
import numpy.matlib,math,random,sys
import numpy as np 

ind = int(sys.argv[1])
snps = int(sys.argv[2])
her = float(sys.argv[4])
d = int(sys.argv[3])
method = int(sys.argv[5])

beta = np.random.normal(0,math.sqrt(her/snps),(snps,1))
env = np.random.normal(0,math.sqrt(1-her),(ind,1))
snps_poss  = [0,1,1,2]
G = numpy.zeros(shape=(ind,snps))

for i in range(ind):
    for j in range(snps):
        G[i][j] = random.choice(snps_poss)

G_mean = G.mean(0)
G_std = np.std(G,0)

G = G-G_mean
G = G/G_std

pheno = np.dot(G,beta)
pheno = pheno+env
pheno_mean = pheno.mean(0)
pheno = pheno-pheno_mean

pheno_file = open("her.pheno.plink","w")
pheno_file.write("FID IID 1\n")

for i in range(pheno.size):
    pheno_file.write(str(i)+" 1 "+str(pheno[i][0])+"\n")
pheno_file.close()

X = G
G = G/math.sqrt(snps) # now the grm is just G*G.t 

if (method == 0) :
    ytKy = (np.linalg.norm(np.dot(G.transpose(),pheno)))**2
    yty = (np.linalg.norm(pheno))**2
    trk2=0
    for _ in range(d):
        z = np.random.normal(0,1,(ind,1))
        trk2 = trk2+(np.linalg.norm(np.dot(G,np.dot(G.transpose(),z))))**2
    trk2 = trk2/d
    #print (ytKy,yty,trk2)
    
else :
    if (method == 1):
        sketching_matrix = np.random.normal(0,1/math.sqrt(snps),(snps,d))
        sk_out = np.dot(G,sketching_matrix)
    elif (method == 2):
        sk_out = G
    
    grm = np.dot(sk_out,sk_out.transpose())
    ytKy = np.dot(pheno.transpose(),np.dot(grm,pheno))[0,0]
    yty = np.dot(pheno.transpose(),pheno)[0,0]
    trk2 = np.sum(np.multiply(grm,grm.transpose())) # trace of grm^2
    #print (ytKy,yty,trk2)
    
new_mat = np.dot(np.matrix([[ind,-ind],[-ind,trk2]]),np.matrix([[ytKy],[yty]]))

#print (new_mat)
print ("heritability is : "+ str ((new_mat[0]/(new_mat[0]+new_mat[1]))[0,0]))
