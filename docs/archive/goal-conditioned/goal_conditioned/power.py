import math
za,zb=1.645,0.842  # one-sided 5%, power 80%
def n_unpaired(p1,p2):
    pb=(p1+p2)/2
    return ((za*math.sqrt(2*pb*(1-pb))+zb*math.sqrt(p1*(1-p1)+p2*(1-p2)))**2)/((p1-p2)**2)
for p1,p2 in [(0.94,0.89),(0.94,0.91),(0.84,0.79),(0.97,0.94),(0.95,0.90)]:
    print('unpaired',p1,p2,round(n_unpaired(p1,p2)))
# paired McNemar: discordant rate under H0 d0 (noise), net shift delta; n = (za*sqrt(d)+zb*sqrt(d-delta^2))^2/delta^2
def n_mcnemar(d,delta):
    return ((za*math.sqrt(d)+zb*math.sqrt(d-delta**2))**2)/delta**2
for d in (0.02,0.05,0.10):
    for delta in (0.02,0.03,0.05):
        print('mcnemar disc',d,'delta',delta,round(n_mcnemar(d+delta,delta)))
# binomial SE at n
for n in (32,64,128,256,512,1024):
    print(n, 'SE@0.94', round(math.sqrt(.94*.06/n),4),'SE@0.5',round(math.sqrt(.25/n),4))
# game hours per update
print('h/update',32768*4/30/3600)
