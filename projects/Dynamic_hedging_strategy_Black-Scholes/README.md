Dynamic Hedging Strategy in the Black and Scholes framework
================
Mamadou KANOUTE
12/11/2020

Les formules utilisées dans cette simulation ont été démontrées dans la
version manuscrite du devoir de maison dans les questions
correspondantes.

### Variables globales

``` r
S0 = 100
sigma = 0.2
mu = 0.03
r = 0.015
K = 97
T_ = 1
tabN = c(10, 100, 500, 1000, 5000, 10000)

M = 100 # Le nombre de simulations indépendantes pour vérifier la convergence de VT et phi(ST)
tabDeviation = c() # Pour afficher l'écart type pour vérifier la convergence
```

**Pour les affichages suivants des différentes fonctions crées, on fait
le test avec N = 10, après on fera pour les différents N evoqués dans le
devoir de maison.**

### Simulation de St

On utilise l’expression suivante pour simuler l’actif risqué par la
formule de Black-Scholes.

![](shoots/discreteSt.png)

La fonction est discreteSt

``` r
discreteSt = function(S0, sigma, N, mu, T_) {
  h = T_/N
  tabSt= c(S0)
  Ni = rnorm(N,0,1)
  for(i in 1:N) {
    tabSt[i+1] = tabSt[i]*exp(sigma*sqrt(h)*Ni[i] + (mu-((sigma^2)/2))*h)
  }
  return (tabSt)  
}
```

**On affiche le tableau contenant S0, Sti pour i = 1, … N pour N = 10**

``` r
discreteStTab = discreteSt(S0, sigma, tabN[1], mu, T_)
print(discreteStTab)
```

    ##  [1] 100.00000  97.37200  95.10656  99.73288  93.89048  95.75747  89.00993
    ##  [8]  83.29097  81.46417  79.64362  78.68495

**Graphique de St POur N =
10**

``` r
plot(seq(0,T_,T_/tabN[1]),discreteStTab,type='l', lwd=2, xlab="La grille de pas T/N", ylab="St")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotSti-1.png)<!-- -->

### Valeur du porte-feuille v(t, St)

En posant ![](shoots/payoff.png),

On obtient la formule suivante demontrée dans mes copies du devoir de
maison,

![](shoots/forall.png), ![](shoots/vt.png) avec ![](shoots/d1_1.png) et
![](shoots/d2_1.png)

Ainsi pour t = 0, ![](shoots/v0.png) avec ![](shoots/d1_2.png) et
![](shoots/d2_2.png)

``` r
d1 = function(T_,t, K, x, sigma) {
  result=(log(x/K)+(r+sigma^2/2)*(T_-t))/(sigma*sqrt(T_-t))
  return (result)
}
Vt = function(T_,t, K, S, sigma) {
  d1_ = d1(T_,t, K, S, sigma)
  d2_ = d1_ - sigma*sqrt(T_-t)
  result=S*pnorm(d1_, 0,1)-K*exp(-r*(T_-t))*pnorm(d2_, 0,1)
  return(result);
}
```

### Calcul du grec ![](shoots/delta0.png)

On utilise la formule suivante demontrée, ![](shoots/delta.png) avec
![](shoots/d1_1.png)

``` r
deltaGrec = function(T_,t, K, S, sigma) {
  d1_ = d1(T_,t, K, S, sigma)
  result=pnorm(d1_, 0,1)
  return(result);
}
```

### Discrétisation de v(t, St)

On utilise la formule de discrétisation de Vt donnée dans l’exercice,
![](shoots/Vt_discret.png) avec ![](shoots/V0_discrete.png)

``` r
discretizationVt = function(S0, discreteStTab, K, T_, r, sigma, N) {
  #N = length(discreteStTab)
  #if (echeanceCall(discreteStTab[1], K) > 0) {
  grid = seq(0,T_,T/N)
  #}
  h = T_/N
  V0 = Vt(T_,0, K, S0, sigma)
  tabV = c(V0)
  #Sti = S0
  for (i in 1:N) {
    delta = deltaGrec(T_,grid[i], K, discreteStTab[i], sigma)
    differenceSt = discreteStTab[i+1] - discreteStTab[i]
    tabV[i+1] = (tabV[i]) + (delta*differenceSt) + (tabV[i]-delta*discreteStTab[i])*(exp(r*h) - 1)
  }
  return(tabV)
}
```

Affichage de la valeur du porte-feuille pour N = 10

``` r
tabV = discretizationVt(S0,discreteStTab, K, T_, r, sigma, tabN[1])
print(tabV)
```

    ##  [1] 10.210259  8.480158  7.108976  9.444256  5.727420  6.547612  3.022243
    ##  [8]  1.360516  1.171974  1.114443  1.114930

**Graphique de Vt pour N =
10**

``` r
plot(seq(0,T_,T_/tabN[1]),tabV,type='l', lwd=2, xlab="La grille de pas T/N", ylab="Vt")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotVti-1.png)<!-- -->

### Vérification de notre discrétisation telle que ![](shoots/convergence.png)

On sait que ![](shoots/conv1.png) quand ![](shoots/conv2.png)

**Fonction echéance**

``` r
echeanceCall = function(x, K) {
  result = c(x-K,0)
  return (max(result))
}
```

**Fonction qui simule indépendamment M fois ![](shoots/simulations.png)
**

``` r
simulation = function(S0,K, T_, r, mu, sigma,N, M) {
  tabSimul = c()
  for(i in (1:M)) {
    discreteStTab = discreteSt(S0, sigma, N, mu, T_)
    val = discretizationVt(S0,discreteStTab, K, T_, r, sigma, N)
    VT =  val[length(val)]
    ST = discreteStTab[length(discreteStTab)]
    tabSimul[i] = VT - echeanceCall(ST, K)
  }
  return(tabSimul)
}
```

**Affichage de M simulations indépendantes pour N = 10**

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[1], M)
print(tabSimul)
```

    ##   [1]  1.00278153 -0.03214169 -1.91252271 -0.19504761 -2.73372816 -2.45560807
    ##   [7] -1.52781778 -2.02625180  2.16884130  0.86932703  0.40099643  0.71092342
    ##  [13] -2.92614685  1.14151633  0.62339625  2.10815105 -3.65873678 -0.13655423
    ##  [19] -1.58622614  0.14654294 -1.69660619  0.99696705  0.23805484  1.95768521
    ##  [25] -0.46375722  0.77191850 -5.07200594 -0.64150050  0.83969456 -0.45492781
    ##  [31]  0.10379656  1.12750915  3.47494478 -0.71958495  1.24399140 -1.85700412
    ##  [37]  0.29903044  0.61473602 -2.28303729 -1.33406460  1.81437447 -2.27969603
    ##  [43] -0.61860635  0.89382797 -2.19321664 -1.08524151  1.17780456 -0.43198060
    ##  [49]  0.20677380 -0.73148704 -1.78021141 -1.06068541  1.60034264 -1.66755409
    ##  [55] -1.54432838  1.08608228  3.23413258  0.30655540 -0.22412313  3.41154873
    ##  [61]  1.46908771  2.01225087 -0.98570282 -0.41183485  1.19661556  2.00357436
    ##  [67]  0.57668159 -1.36542752  0.86004389 -5.19632129  4.90654975  1.83166913
    ##  [73] -0.81354681  2.49879060  2.32119204 -0.56693909 -0.62492450 -1.24397433
    ##  [79] -1.99415812  0.42784280 -2.14011040 -0.37535899 -0.86089660 -4.49496675
    ##  [85] -0.31836481  0.55993011 -0.66059468  3.27865652  1.66407341 -2.02837603
    ##  [91]  2.22445275 -2.59969749 -3.26155504  0.54134648  2.00566289 -2.48501704
    ##  [97]  0.54747182  0.69019142 -0.68539107  0.73023629

**Graphique des M simulations indépendantes pour N =
10**

``` r
plot(seq(1,M,1),tabSimul,type='l', lwd=2, xlab="M valeurs", ylab="simulations")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotTabSimul-1.png)<!-- -->

**Deviation**

``` r
deviation = function(X) {
  return(sd(X))
}
```

**Affichage de l’écart-type**

``` r
devi = deviation(tabSimul)
print(devi)
```

    ## [1] 1.859806

``` r
tabDeviation[1] = devi
```

**C’est normal car N n’est pas assez grand pour que
![](shoots/conv1.png)** .

## Test pour différents N

**On remarque que plus N est grand, ![](shoots/simulations.png) converge
vers 0 avec M = 100 simulations indépendantes de
![](shoots/diffSimul.png)**

Ce qui est vérifié, **en effet ![](shoots/conv1.png) lorsque
![](shoots/conv2.png)**

#### Pour N = 10

Déjà fait en haut

#### Pour N = 100

On discrétise St

``` r
discreteStTab = discreteSt(S0, sigma, tabN[2], mu, T_)
```

**Graphique de St Pour N =
100**

``` r
plot(seq(0,T_,T_/tabN[2]),discreteStTab,type='l', lwd=2, xlab="La grille de pas T/N", ylab="St")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotSti2-1.png)<!-- -->

On discrétise Vt

``` r
tabV = discretizationVt(S0,discreteStTab, K, T_, r, sigma, tabN[2])
```

**Graphique de Vt pour N =
100**

``` r
plot(seq(0,T_,T_/tabN[2]),tabV,type='l', lwd=2, xlab="La grille de pas T/N", ylab="Vt")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotVti2-1.png)<!-- -->

On fait M simulations de ![](shoots/diffSimul.png)

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[2], M)
```

**Affichage des M simulations indépendantes pour N = 100**

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[2], M)
print(tabSimul)
```

    ##   [1]  0.29546773 -0.45906596 -0.23430150  0.28097150  0.08121827 -0.26816566
    ##   [7]  0.68570843  0.21863457 -0.05787404 -0.84881432  0.03091214  0.34372526
    ##  [13] -0.55012316  0.42330952 -0.06741249 -1.24913734 -0.76275011 -1.40609705
    ##  [19] -0.45755591  0.12097156 -1.18769893 -0.23035638  0.08536542 -0.45349863
    ##  [25] -1.28204168  0.12685006  0.12636061 -0.62075991 -0.06117934 -0.55248350
    ##  [31]  0.12266995 -1.54572322 -0.47178533 -0.44557316  0.13393695  0.08417791
    ##  [37]  0.87378930 -0.02891734 -0.00080080  0.33792648 -0.17206381 -1.15244259
    ##  [43]  0.12557706 -0.74334986 -0.15956895 -0.79864181 -0.41982008 -0.26904369
    ##  [49]  0.30870772  0.00868481 -0.58214700 -0.17174296  1.49911149 -0.27920138
    ##  [55] -0.04609099  0.13125487  0.10328158  1.47186430  1.20545556  0.52124440
    ##  [61]  0.30448813  1.46875423  1.17984777 -0.72159408  0.19472885  0.32231309
    ##  [67]  0.48808808 -0.70666839 -0.13027890 -0.02247798 -0.10161250  0.59662085
    ##  [73] -0.84974619  0.21765002  0.05991700  1.35224488 -2.95793700 -0.74177089
    ##  [79] -0.14743361 -0.59852034 -0.35116005 -0.87208990 -0.90463957  0.49837857
    ##  [85]  0.10611357  0.66386868 -0.42351134  0.58171909  0.58276654 -1.52723432
    ##  [91] -2.55284251 -1.15512636 -1.58810874 -0.59550742  0.33429721  0.04619766
    ##  [97] -0.54292936  1.09972559 -0.38781140 -0.26945124

**Graphique des M simulations indépendantes pour N =
100**

``` r
plot(seq(1,M,1),tabSimul,type='l', lwd=2, xlab="M valeurs", ylab="simulations")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotTabSimul2-1.png)<!-- -->

**Affichage de l’écart-type**

``` r
devi = deviation(tabSimul)
print(devi)
```

    ## [1] 0.7602731

``` r
tabDeviation[2] = devi
```

#### Pour N = 500

On discrétise St

``` r
discreteStTab = discreteSt(S0, sigma, tabN[3], mu, T_)
```

**Graphique de St Pour N =
500**

``` r
plot(seq(0,T_,T_/tabN[3]),discreteStTab,type='l', lwd=2, xlab="La grille de pas T/N", ylab="St")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotSti3-1.png)<!-- -->

On discrétise Vt

``` r
tabV = discretizationVt(S0,discreteStTab, K, T_, r, sigma, tabN[3])
```

**Graphique de Vt pour N =
500**

``` r
plot(seq(0,T_,T_/tabN[3]),tabV,type='l', lwd=2, xlab="La grille de pas T/N", ylab="Vt")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotVti3-1.png)<!-- -->

On fait M simulations de ![](shoots/diffSimul.png)

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[3], M)
```

**Affichage des M simulations indépendantes pour N = 500**

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[3], M)
print(tabSimul)
```

    ##   [1] -0.123323284 -0.020515879  0.302752530  0.043126885 -0.659443275
    ##   [6] -0.117190954 -0.179893976  0.361224067  0.143044610  0.038035649
    ##  [11]  0.036957990 -0.511110665  0.613319467  0.339291158  0.622101696
    ##  [16] -0.016310711 -0.001897113  0.222684106  0.096846070 -0.452515486
    ##  [21]  0.016039451 -0.234616252 -0.291150280 -0.387171765  0.084962108
    ##  [26] -0.008027041 -0.156917622  0.072440126  0.072903320 -0.023094325
    ##  [31]  0.301924031  0.420090117  0.201262959 -0.128100634  0.018049766
    ##  [36]  0.218425769  0.700527908  0.082604981  0.741054323  0.269012367
    ##  [41]  0.257325069  0.325013501 -0.238836338 -0.423615600 -0.197382148
    ##  [46] -0.112863005  0.026517693 -0.049384771 -0.025485074  0.143537180
    ##  [51] -0.164274774  0.058922967  0.174731152 -0.147833472 -0.057914596
    ##  [56]  0.157238406  0.145200303 -0.232684633  0.015459043  0.152588859
    ##  [61] -0.244107564  0.226089861  0.059144863  0.157158727  0.222419255
    ##  [66]  0.241362754  0.258880189  0.004899532  0.043301629 -0.313579475
    ##  [71] -0.190154830  0.606655938 -0.083320742 -0.064891378  0.033755239
    ##  [76] -0.406029643 -0.055162980 -0.036071740  0.102856964  0.014793415
    ##  [81] -0.850366364 -0.196418468  0.593415212 -0.279077701 -0.645176170
    ##  [86] -0.290271404 -0.002875016 -0.006871347 -0.245150316 -0.242665630
    ##  [91]  0.006463882 -0.198224703  0.101844062 -0.141470873  0.043973253
    ##  [96]  0.334135331 -0.021027213 -0.006365911  0.530437312  0.280506394

**Graphique des M simulations indépendantes pour N =
500**

``` r
plot(seq(1,M,1),tabSimul,type='l', lwd=2, xlab="M valeurs", ylab="simulations")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotTabSimul3-1.png)<!-- -->

**Affichage de l’écart-type**

``` r
devi = deviation(tabSimul)
print(devi)
```

    ## [1] 0.2842173

``` r
tabDeviation[3] = devi
```

#### Pour N = 1000

On discrétise St

``` r
discreteStTab = discreteSt(S0, sigma, tabN[4], mu, T_)
```

**Graphique de St Pour N =
1000**

``` r
plot(seq(0,T_,T_/tabN[4]),discreteStTab,type='l', lwd=2, xlab="La grille de pas T/N", ylab="St")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotSti4-1.png)<!-- -->

On discrétise Vt

``` r
tabV = discretizationVt(S0,discreteStTab, K, T_, r, sigma, tabN[4])
```

**Graphique de Vt pour N =
1000**

``` r
plot(seq(0,T_,T_/tabN[4]),tabV,type='l', lwd=2, xlab="La grille de pas T/N", ylab="Vt")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotVti4-1.png)<!-- -->

On fait M simulations de ![](shoots/diffSimul.png)

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[4], M)
```

**Affichage des M simulations indépendantes pour N = 1000**

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[4], M)
print(tabSimul)
```

    ##   [1]  0.215324827  0.255297243  0.237009616  0.453609020  0.010573473
    ##   [6]  0.221625503  0.110663406 -0.074752259 -0.243323979  0.676861783
    ##  [11]  0.111383484 -0.220556157  0.068692271  0.123793299 -0.191271227
    ##  [16] -0.020547729  0.047795171  0.193243128 -0.086952321  0.132737922
    ##  [21]  0.363378905 -0.016789159 -0.064848961  0.148097219 -0.037743843
    ##  [26]  0.015665849  0.048043653  0.082000706 -0.151199077 -0.091215260
    ##  [31]  0.188098378 -0.085410831 -0.092603838 -0.137460994  0.228710921
    ##  [36] -0.371113936 -0.044511520  0.296557429 -0.226127503  0.234816449
    ##  [41]  0.110459098  0.001915939 -0.136918530  0.036950934  0.226977753
    ##  [46] -0.302478802  0.089134537  0.041070112 -0.054021958 -0.223121403
    ##  [51]  0.076852919 -0.044072958 -0.139936236 -0.163954215 -0.071104090
    ##  [56]  0.048153655  0.115228777  0.179370900  0.340668949  0.162543722
    ##  [61]  0.053114701 -0.005918547  0.246510564  0.133720331  0.467655824
    ##  [66]  0.144901439  0.097841173 -0.027872513  0.244586773 -0.071460217
    ##  [71]  0.182198604  0.431322895 -0.386346061 -0.154653239  0.335842858
    ##  [76]  0.030842547 -0.155743852  0.217258059  0.078345204 -0.335474941
    ##  [81]  0.073544972 -0.407052548  0.315759308  0.134254972 -0.154774595
    ##  [86]  0.226836053 -0.005558121  0.108089770 -0.108614709  0.035811036
    ##  [91]  0.023328326  0.080114190 -0.080167323 -0.451478772 -0.293292399
    ##  [96]  0.032901313  0.078735669  0.170646897 -0.257294871  0.251952771

**Graphique des M simulations indépendantes pour N =
1000**

``` r
plot(seq(1,M,1),tabSimul,type='l', lwd=2, xlab="M valeurs", ylab="simulations")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotTabSimul4-1.png)<!-- -->

**Affichage de l’écart-type**

``` r
devi = deviation(tabSimul)
print(devi)
```

    ## [1] 0.2031

``` r
tabDeviation[4] = devi
```

#### Pour N = 5000

On discrétise St

``` r
discreteStTab = discreteSt(S0, sigma, tabN[5], mu, T_)
```

**Graphique de St Pour N =
5000**

``` r
plot(seq(0,T_,T_/tabN[5]),discreteStTab,type='l', lwd=2, xlab="La grille de pas T/N", ylab="St")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotSti5-1.png)<!-- -->

On discrétise Vt

``` r
tabV = discretizationVt(S0,discreteStTab, K, T_, r, sigma, tabN[5])
```

**Graphique de Vt pour N =
5000**

``` r
plot(seq(0,T_,T_/tabN[5]),tabV,type='l', lwd=2, xlab="La grille de pas T/N", ylab="Vt")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotVti5-1.png)<!-- -->

On fait M simulations de ![](shoots/diffSimul.png)

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[5], M)
```

**Affichage des M simulations indépendantes pour N = 5000**

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[5], M)
print(tabSimul)
```

    ##   [1] -0.019910162 -0.029256411 -0.018123208 -0.077180382 -0.072461710
    ##   [6] -0.013667991 -0.064943985  0.053742874  0.017963088 -0.185300816
    ##  [11]  0.046354150 -0.050399863 -0.012323317 -0.017646919  0.122862409
    ##  [16] -0.004601627  0.093343523  0.055698392  0.141676550 -0.051069224
    ##  [21] -0.039069493 -0.066030388  0.053511836 -0.015581704  0.002202825
    ##  [26] -0.030642164 -0.056217919 -0.179753411  0.008792382  0.025206396
    ##  [31]  0.093423246 -0.093099095  0.033809383 -0.071218529  0.082489079
    ##  [36] -0.119877775 -0.020304596  0.145889032  0.030952039  0.037150948
    ##  [41] -0.077304169 -0.040558155  0.179146625 -0.029799190 -0.129625226
    ##  [46] -0.104453239  0.027139866  0.007344559 -0.003381924 -0.035572536
    ##  [51]  0.130090245  0.088790331  0.070121983  0.003254272  0.061939098
    ##  [56] -0.072006304  0.014506461  0.066239432  0.101299381  0.028912898
    ##  [61] -0.119749149  0.079681230 -0.132564048  0.005043977 -0.082106357
    ##  [66] -0.122975774 -0.004155854 -0.221750879 -0.003580392 -0.073653560
    ##  [71]  0.038579276  0.031628552 -0.144138204 -0.002844276 -0.114654621
    ##  [76]  0.027444362  0.026320430  0.040915947  0.027059252  0.062675722
    ##  [81]  0.008199214  0.070875377 -0.028509174  0.025097315 -0.023794555
    ##  [86] -0.183529254  0.070962166  0.044740280 -0.004422657  0.198870813
    ##  [91]  0.054906890  0.134703357  0.116665013  0.120141694  0.162469947
    ##  [96] -0.116006886  0.162572163  0.028956833 -0.137894569 -0.230273865

**Graphique des M simulations indépendantes pour N =
5000**

``` r
plot(seq(1,M,1),tabSimul,type='l', lwd=2, xlab="M valeurs", ylab="simulations")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotTabSimul5-1.png)<!-- -->

**Affichage de l’écart-type**

``` r
devi = deviation(tabSimul)
print(devi)
```

    ## [1] 0.08897118

``` r
tabDeviation[5] = devi
```

#### Pour N = 10000

On discrétise St

``` r
discreteStTab = discreteSt(S0, sigma, tabN[6], mu, T_)
```

**Graphique de St Pour N =
10000**

``` r
plot(seq(0,T_,T_/tabN[6]),discreteStTab,type='l', lwd=2, xlab="La grille de pas T/N", ylab="St")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotSti6-1.png)<!-- -->

On discrétise Vt

``` r
tabV = discretizationVt(S0,discreteStTab, K, T_, r, sigma, tabN[6])
```

**Graphique de Vt pour N =
10000**

``` r
plot(seq(0,T_,T_/tabN[6]),tabV,type='l', lwd=2, xlab="La grille de pas T/N", ylab="Vt")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotVti6-1.png)<!-- -->

On fait M simulations de ![](shoots/diffSimul.png)

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[6], M)
```

**Affichage des M simulations indépendantes pour N = 10000**

``` r
tabSimul = simulation(S0,K, T_, r, mu, sigma,tabN[6], M)
print(tabSimul)
```

    ##   [1]  0.013511343 -0.016154835 -0.042929567 -0.030012321  0.050337265
    ##   [6] -0.070389534  0.039818785 -0.097956226 -0.030729179  0.027906196
    ##  [11]  0.044684481  0.009628240 -0.131324815  0.038645183 -0.005451663
    ##  [16] -0.015017303 -0.032405009 -0.043153185 -0.017256668 -0.038935227
    ##  [21] -0.106752555 -0.054984551 -0.095842961 -0.116718054 -0.097021403
    ##  [26] -0.109761810  0.037789096 -0.053806507 -0.041366462  0.052421943
    ##  [31]  0.015340688 -0.026344675 -0.012685900 -0.057870492  0.039901377
    ##  [36]  0.005824892 -0.113980746  0.006530601  0.079525325 -0.008311844
    ##  [41] -0.003870050 -0.042590279 -0.052838177  0.004569343 -0.094331247
    ##  [46]  0.137205591 -0.041741936  0.091630005 -0.037313117  0.003379664
    ##  [51]  0.111733859  0.018190501  0.107354444  0.017954043 -0.093108099
    ##  [56] -0.023107612  0.086325408  0.013500232 -0.011921442  0.115786795
    ##  [61] -0.168246535 -0.004182835 -0.036248127 -0.072511609 -0.041502804
    ##  [66]  0.152292792  0.064983540  0.039317689  0.037392674 -0.007064708
    ##  [71] -0.068794311  0.126051931 -0.004727940  0.006511622  0.018063191
    ##  [76]  0.017243494 -0.058856501  0.132652738  0.056094054  0.064772349
    ##  [81]  0.028372812  0.019555915  0.136344750 -0.038595320 -0.031419879
    ##  [86]  0.114319233  0.002371132 -0.068077685  0.053441331 -0.065481659
    ##  [91]  0.005985657  0.016500272 -0.013204730 -0.045650703 -0.023000266
    ##  [96]  0.057872597 -0.077403824  0.017676552  0.039590988  0.008283587

**Graphique des M simulations indépendantes pour N =
10000**

``` r
plot(seq(1,M,1),tabSimul,type='l', lwd=2, xlab="M valeurs", ylab="simulations")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotTabSimul6-1.png)<!-- -->

**Affichage de l’écart-type**

``` r
devi = deviation(tabSimul)
print(devi)
```

    ## [1] 0.06502502

``` r
tabDeviation[6] = devi
```

## Affichage des ecart-types précédents pour M simulations indépendantes en fonction de N

``` r
#len = length(tabDeviation)
logTabDeviation = log(tabDeviation + 1)
Nmax = max(tabN)
plot(tabN,logTabDeviation,type='l', lwd=2, xlab="N", ylab="ecart-type")
```

![](Mamadou_KANOUTE_M2_DS_files/figure-gfm/Sti%20plotAllEcartypes-1.png)<!-- -->

**On remarque que plus N est grand, on a une decroissance assez
importante de l’ecart-type**
