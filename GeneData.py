import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.linalg as la
from scipy.stats import entropy
import math
import IPython
import time
import importlib
import io
#import cute; importlib.reload(cute); # force-reload custom package(s)
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from itertools import compress

def Load_Dataset(filename:str): 
    """ Read and load RNA-Seq gene expression data in a CSV format, and return a numpy ndarray of data having fields with different datatypes and can be accessed with their name"""
    
    with open(filename, newline='') as csvfile:
        data = []
        reader = csv.reader(csvfile) #reader object iterates through the csv file
    
        next(reader)                   # print the header line
        #print(next(reader))
        for row in reader:
            data.append (row)           #loading data 

    data = np.array(data)
    #data = np.where(data =='NA',0.00, data)  #assign the floating point number to places where it says 'NA'  in the dataset                                 
    data = np.delete(data,  np.where(data[:,3] =='NA'),  axis = 0)  #delete rows where the expression value =='NA'
    
    times = data[:,1].astype(float)     # times is the first column of data
    genes = data[:,2]                   # strings
    exprs = data[:,3].astype(float)     # expressions 
    sds = data[:,4].astype(float)       # standarddeviation
    conds = data[:,5]                   # 'GCSF-OHT' or 'IL3'
   
 #structured datatype having fields with different datatypes and can be accessed with their name
    
    types = np.dtype([('timepoints', times.dtype),        
                      ('geneNames', genes.dtype),
                      ('expression', exprs.dtype),
                      ('sd', sds.dtype),
                      ('condition', conds.dtype)
                     ]
                    )           
    
    data_typed = np.empty(len(times), dtype=types)  #returns an array of uninitialized data with the given size and given datatype
    data_typed['timepoints'] = times                #assigning values in the dataset to the specified fields 
    data_typed['geneNames'] = genes
    data_typed['expression'] = exprs
    data_typed['sd'] = sds
    data_typed['condition'] = conds
                      
    
    
    return data_typed
    
 

def ReadCSV(filename:str):
    '''Read and load data in a CSV file as a numpy ndarray'''
    
    with open(filename, newline='') as csvfile:
        Data = []
        reader = csv.reader(csvfile) #reader object iterates through the csv file
    
        next(reader)                   # print the header line
        #print(next(reader))
        for row in reader:
            Data.append (row)           #loading data 

    Data = np.array(Data)
    Data = Data.astype(float)
    #Data = np.delete(Data,  np.where(Data[:,1:] =='NA'),  axis = 0)  #delete rows where the expression value =='NA'
    
    return Data

def ReadGeneNames(filename: str):
    """Read only the first column (gene names) from a CSV file."""
    gene_names = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gene_names.append(row)  # take only the first column
    return gene_names

def geneData(geneName:str , condition:str , data):
    """ Gene Expression values and time points for a given gene name with the given condition"""
    
    g = data['geneNames'] == geneName    #return an array of gene names that is equal to the given gene name
    c = data['condition'] == condition   #return an array of conditions that is equal to the given condition
    
    x = np.logical_and(g , c)            #return an array of values with the given gene name and condition
    
    timePoints = data['timepoints'][x]   #assign time values of x in the dataset to the variable 'timePoints'
    exprsValues = data['expression'][x]  #assign expression values of x in the dataset to the variable 'exprsValues' 
       
    XX = np.column_stack((timePoints , exprsValues))      #stack two 1-D arrays timePoints and exprsValues, into a 2D array
    return XX




def readGeneList(listOfGeneNames:list):
    """ Reading Gene Names listed in a text file """
    
    f = open(listOfGeneNames ,"r")
    s = f.read()                         # string
    geneNameList = s.split() 
    
    return geneNameList


def exprs(gList:list, condition:str , data, subset:bool):
    """ Returns timepoints and expression values for a given list of genes with the condition"""
    
    # Determine number of genes
    #print(gList.__class__)
    numGenes = len(gList)
    
    # Determine number of timepoints
    g = data['geneNames'] == gList[0]
    c = data['condition'] == condition
    x = np.logical_and(g , c)
    timePoints = data['timepoints'][x]
    numTimepoints = len(timePoints)
    timePoints = timePoints.reshape(numTimepoints,1)
    
    # Initialize numpy array
    exprsValues = np.zeros((numTimepoints, numGenes))
   
    Gene_counter = 0

    if subset:
        for i in gList:
            g = data['geneNames'] == i
            x = np.logical_and(g , c)
            exprsValues[:,Gene_counter] = data['expression'][x]
            Gene_counter += 1
    else:
        exprsValues = data['expression'][c].reshape(numTimepoints, numGenes)
    
    return timePoints , exprsValues 


def scaleGeneExpression(Xgt):
    """Scales gene expression so that each gene (assumed to be rows) has maximum expression of one over all samples/conditons (assumed to tbe columns)"""

    normExpr = np.zeros(Xgt.shape)
    for j in range(Xgt.shape[0]):
        normExpr[j,:] = Xgt[j,:]/np.max(Xgt[j,:])

    return normExpr

def normalizedExprs(exprsValues):
    """Returns normalized expression values as a numpy ndarray for a given list of genes with the condition.
        Input variable should be a numpy ndarray  """
    
    numgenes = exprsValues.shape[1]               #defines the dimension of the number of genes
    normalizedExprs = np.zeros(exprsValues.shape) #initialize a numpy array with the dimension of expression values for one gene at a time
    
    #getting normalized expressions for one gene for all time points and repeating the same for all the other genes
    for i in range(0,numgenes):
        normalizedExprs[:,i] = exprsValues[:,i] / (max(exprsValues[:,i]) + 1e-300)
        
        
    return  normalizedExprs

def Normalized(gcsfExpr , il3Expr):
    '''Returns normalized expressions as a tuple over both conditions. 
         Input variables should be numpy ndarrays'''
    
    numtimepoints = gcsfExpr.shape[0]
    combinedExpr = np.concatenate((gcsfExpr, il3Expr), 0)
    normcombined = normalizedExprs(combinedExpr)
    
    return normcombined[0:numtimepoints,:] , normcombined[numtimepoints:,:]
    
def HeatMap(timePoints , exprsValues , geneNameList):
    """ Plotting heat map of genes for a particular condition """
    
    fig, ax = plt.subplots()
    im = ax.imshow(exprsValues)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(geneNameList)), labels=geneNameList , fontsize = 10)
    ax.set_yticks(np.arange(len(timePoints)), labels=timePoints , fontsize = 10)

    # Rotate the tick labels and set their alignment.
    Plot = plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
    
    fig.colorbar(im, ax=ax, label='colorbar' )
    fig.set_figwidth(30)
    fig.set_figheight(10)
    
    return Plot



def HeatMapBoth(timePointsGCSF , timePointsIL3 , normalizedBoth1 ,geneNameList):
    '''Plot both heatmaps for the GCSF and IL3 treatments with normalized expression values'''
    
    HeatMapGCSF = HeatMap(timePointsGCSF , normalizedBoth1[0] , geneNameList)
    HeatMapIL3 = HeatMap(timePointsIL3 , normalizedBoth1[1] , geneNameList)
    
    return HeatMapGCSF , HeatMapIL3



# YLL 2024-6:
# - you'll want to have code where you can use "fixed seed" or "clock-based seed"
# - note that in the NMF code, if you call np.random.seed(1234), then
#    EVERTy SUBSEQUENT call to np.random.rand in the current session will be predictable.
# - Really, in the code below, if you want to use a user-supplied seed,
# you should contruct a local instance of a RNG  using rng = np.random.default_rng(seed=1234)
# to avoid insidious side-effects


def NMF(X, K, tol):   # def NMF(X, K, tol, seed):
    """ NMF Algorithm : X must be an numpy ndarray which will be factorized to K dimensions with tolerance = tol. Returns W, H matrices and Frobenius Norm """

    
    FrobNormPrev = 0 # Do determine change in Frobenius norm for stopping
    N,M = X.shape  # get N and M from X, which is a NxM array
    W = np.random.rand (N,K)  # initialize W to be a random NxK array     # ...(N,K,seed=seed
    H = np.random.rand (K,M)  
    
    assert not np.isnan(np.sum(X)), 'ERROR! ARRAY CONTAINS NaN(s)!'
    assert not np.isnan(np.sum(W)), 'ERROR! ARRAY CONTAINS NaN(s)!'
    assert not np.isnan(np.sum(H)), 'ERROR! ARRAY CONTAINS NaN(s)!'
    
    history = []
    iter = 0
    
    while (True):
        #start = time.time()
        iter = iter+1
        
    #------- update H
        tWX = np.dot(W.T, X)    # multiplicative updates   # actually WtX
        WtWH = np.dot(W.T.dot(W), H)
        Hnew = H * tWX
        Hnew /= WtWH + 1e-16

        assert not np.isnan(np.sum(Hnew)), 'ERROR! ARRAY CONTAINS NaN(s)!'
        
        #--- calculate delta which is difference between UPDATED value of H and the value BEFORE THE UPDATE.
        deltaH = la.norm(Hnew - H)  
        H = Hnew
        
    #-------- update W
        XH = X.dot(H.T)
        WHtH = W.dot(H.dot(H.T))   # actually WHHt
        Wnew = W * XH
        #assert np.all(WHtH>0)
        Wnew /= WHtH + 1e-16
        deltaW = la.norm(Wnew - W)    
        assert not np.isnan(np.sum(Wnew)), 'ERROR! ARRAY CONTAINS NaN(s)!'

        W = Wnew   # update variable W (overwriting previous contents)
        
    #-------- print Frobenius norm using Alternate Gradient Descent
        WH = W.dot(H)
        FrobNorm = la.norm(X - WH)
               
    #-------- print Kullback-Leibler divergence
        #KLDiv = entropy(X, WH).sum() - (X - WH).sum() 
    
        #print("deltaH = ",deltaH,"\t", "deltaW=",deltaW," \t","||X-WH|| = ", FrobNorm,"\t","KLDiv(X||WH)=", KLDiv)
        
#        if deltaH < tol and deltaW < tol :##and (la.norm(X-WH) < tol):
#            break

        history.append (FrobNorm)
        if iter%1==0:
            IPython.display.clear_output()
            print (f'iteration={iter}   FrobNorm={FrobNorm}')          

        if (abs(FrobNorm - FrobNormPrev) < tol):
            break
        
        FrobNormPrev = FrobNorm
        #end = time.time()
        #print(end - start)       
    return W,H,history



def X_outputs(k , tol , X , normalizedBoth ):

    
    w_both,h_both,History_both = NMF(X, k , tol) 
    plt.plot(History_both) ; plt.ylim(bottom=0)

    # w_gcsf,h_gcsf,History_gcsf = NMF(normalizedBoth1[0].T, k , tol)
    # w_il3,h_il3,History_il3 = NMF(normalizedBoth1[1].T, k , tol)
    # fig, axs = plt.subplots(1,2)
    # axs[0].plot(History_gcsf)
    # axs[0].set_title("GCSF" , fontsize = 10)
    # axs[1].plot(History_il3)
    # axs[1].set_title("IL3" , fontsize = 10)
    # plt.show()   

    #X = combinedConditions1
    # X_gcsf = normalizedBoth1[0].T
    # X_il3 = normalizedBoth1[1].T
    Xapprx_both = np.dot(w_both , h_both) #NMF ran only for for both conditions
    # Xapprx_gcsf = np.dot(w_gcsf , h_gcsf) #NMF ran only for gcsf condition
    # Xapprx_il3 = np.dot(w_il3 , h_il3)    #NMF ran only for il3 condition

    #return X, X_gcsf, X_il3, Xapprx_both, Xapprx_gcsf, Xapprx_il3, h_both, h_gcsf, h_il3
    return X, Xapprx_both,  h_both


def filter_low_expressing_genes(Xgt, geneNames, threshold):
    """ Take X matrix, the list of gene names, and a threshold value as
    inputs. Filter out genes whose maximum expression across all samples is
    lower than the threshold value. Assumes that rows are genes and columns
    are samples. The outputs are the filtered X matrix and filtered gene
    names """

    # Creating a boolean vector for genes that pass filter
    filt = np.max(Xgt, 1) > threshold

    # Filter Xgt matrix 
    filtXgt = Xgt[filt, :]

    #Filter gene names having expresion greater than the threshold value
    #geneNamesArr = np.array(geneNames).reshape(filt.shape)
    #filtGeneNamesArr = geneNamesArr[filt]
    #filtGeneNames = filtGeneNamesArr.tolist()
    filtGeneNames = list(compress(geneNames, filt))

    return filtXgt, filtGeneNames

def plot_for_one_gene_in_a_list(gene_index, gene_name, gcsf_timepoints_of_gene, il3_timepoints_of_gene, gcsf_exprs_of_gene, il3_exprs_of_gene):
    """ Returns plots for gene expression over timepoints for both condition for one gene at a time. Inputs: index of the gene in the gene list, rows corresponding to gcsf and il3 timepoints and expressions for that particular gene """

    #Consider the maximum expression for both conditions of each gene, to set the same y-axis limit for both plots for each gene. 
    max_gcsf_exprs_of_gene = max(gcsf_exprs_of_gene)
    max_il3_exprs_of_gene = max(il3_exprs_of_gene)
    highest_exprs = max(max_gcsf_exprs_of_gene, max_il3_exprs_of_gene)
    
    fig, axs = plt.subplots(1,2)

    plt.sca(axs[0])   
    plt.plot(gcsf_timepoints_of_gene , gcsf_exprs_of_gene , label = 'X_gcsf')
    plt.legend()
    axs[0].set_title("GCSF")
    plt.ylim(top = (highest_exprs + highest_exprs * 0.05))

    plt.sca(axs[1])  
    plt.plot(il3_timepoints_of_gene , il3_exprs_of_gene , label = 'X_il3')
    plt.legend()
    axs[1].set_title("IL3")
    plt.ylim(top = (highest_exprs + highest_exprs * 0.05))

    fig.supxlabel('timepoints' , fontsize = 10)
    fig.supylabel('normalized expressions' , fontsize = 10)
    fig.suptitle(gene_name, fontsize =20)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    plt.show() 

def plots_selected_genes(gcsf, il3, gene_list:list):
    """Calls the above function which plots gene expression over timepoints for one gene. Returns plots of gene expression over timepoints for all genes included in the gene_list. Here the inputs gcsf and il3 are tuples of in which rows are timepoints and columns are expressions of genes included in the gene_list."""
    
    for gene_index in range(len(gene_list)):
        plots = plot_for_one_gene_in_a_list(gene_index, gene_list[gene_index], gcsf[0], il3[0], gcsf[1].T[gene_index], il3[1].T[gene_index])
    
    return plots




def plot_genes(i, X, Xapprx_both,  gcsf, il3 , geneNameList ):
    
    fig, axs = plt.subplots(1,2)

    #print(X.shape)
    # plt.sca(axs[0,0])   
    # plot_gcsfOriginal = plt.plot(gcsf[0] , X_gcsf[i] , label = 'X_gcsf ')
    # plot_gcsfApprx = plt.plot(gcsf[0] , Xapprx_gcsf[i] , label = 'Xapprx_gcsf')
    # plt.legend()
    # axs[0,0].set_title("GCSF")

    # plt.sca(axs[0,1])  
    # plot_il3Original = plt.plot(il3[0] , X_il3[i] , label = 'X_il3')
    # plot_il3Apprx = plt.plot(il3[0] , Xapprx_il3[i] , label = 'Xapprx_il3')
    # plt.legend()
    # axs[0,1].set_title("IL3")

    plt.sca(axs[0])
    #timepoints = np.vstack((gcsf[0] , il3[0]))
    plot_gcsfBoth_Original = plt.plot(gcsf[0] , X[i,0:29] , label = 'X_gcsf')
    plot_gcsfBoth_Apprx = plt.plot(gcsf[0] , Xapprx_both[i,0:29] , label = 'Xapprx_both')
    plt.legend()
    axs[0].set_title("Both GCSF")

    plt.sca(axs[1])
    timepoints = np.vstack((gcsf[0] , il3[0]))
    plot_il3Both_Original = plt.plot(il3[0], X[i, 29:55] , label = 'X_il3')
    plot_il3Both_Apprx = plt.plot(il3[0], Xapprx_both[i,29:55] , label = 'Xapprx_both')
    plt.legend()
    axs[1].set_title("Both IL3")
    
    fig.supxlabel('timepoints' , fontsize = 20)
    fig.supylabel('normalized expressions' , fontsize = 20)
    
    #fig.suptitle(geneNameList.__getitem__, fontsize =30)
    #fig.suptitle(geneName[i], fontsize =35)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    plt.show() 

    return plot_gcsfBoth_Original, plot_gcsfBoth_Apprx, plot_il3Both_Original, plot_il3Both_Apprx  


def plots(Xval , indexList , gcsf , il3 , geneNameList):
    for i in np.ndenumerate(indexList):
        plots = plot_genes(i[1], Xval[0], Xval[1], gcsf, il3 , geneNameList)
    return plots

    
    
def metagene_prof(i, gcsf, il3, h_both):    
    fig, axs = plt.subplots(1,2)
    
    # #NMF done separately over two conditions
    # axs[0,0].plot(gcsf[0] , h_gcsf.T, 'o-' ); axs[0,0].set_xlim (-48, 168)
    # axs[1,0].plot(il3[0] , h_il3.T ,  's-' ); axs[1,0].set_xlim (-48, 168)
    # axs[0,0].set_title("Weights for only GCSF treatment")
    # axs[1,0].set_title("Weights for only IL3 treatment")
    
    #NMF done together for both conditions
    
    axs[0].plot(gcsf[0] , h_both.T[:29,:i], 'o-' ); axs[0].set_xlim (-48, 168)
    axs[1].plot(il3[0] , h_both.T[29:,:i] ,  's-'); axs[1].set_xlim (-48, 168)
    axs[0].set_title("Weights for GCSF(NMF ran for both conditions) ")
    axs[1].set_title("Weights for IL3(NMF ran for both conditions)")
         
    fig.supxlabel('timepoints' , fontsize = 15)
    fig.supylabel('Metagenes normalized Expression profile' , fontsize = 15)
    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.show() 


def NMF_Results(k, iter , tol , X ):
    '''Running NMF for a given number of metagenes with a given tolerance value for given number of iterarions. Returns approximate values of X matrix,  '''
    
    Xapprx_both_list = []
    w_both_list = []
    h_both_list = []
    History_both_list = []
    
    for i in range(iter):
        w_both,h_both,History_both = NMF(X, k , tol) 
        #X = combinedConditions1
        Xapprx_both = np.dot(w_both , h_both) 
        
        Xapprx_both_list.append(Xapprx_both)
        w_both_list.append(w_both)
        h_both_list.append(h_both)
        History_both_list.append(History_both)
        
    return Xapprx_both_list, w_both_list, h_both_list, History_both_list


def Frob_iter(X, Xapprx_both_list, iter):
    '''Computing Frobenius norm among the Xapprx values returned in the function NMF_Results as Frob_norm, and frobenius norm between the original X matrix and approximate X matrix as Frob_norm_X_Xapprx.  Writing the output to a csv file'''
    
    Frob_norm = np.zeros((iter,iter))
    Frob_norm_X_Xapprx = np.zeros((iter,1))
    for i in range(iter):
        for j in range(iter):
       
            Frob = la.norm(Xapprx_both_list[i] - Xapprx_both_list[j])
            Frob_norm[i,j] = Frob
    
    for i in range(iter):
        Frob_X_Xapprx = la.norm(X - Xapprx_both_list[i])
        Frob_norm_X_Xapprx[i] = Frob_X_Xapprx
    #print(Frob_norm_X_Xapprx)
    
    # with open(filename , 'w',  newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["1","2","3","4","5","6","7","8","9","10"])
    #     writer.writerows(Frob_norm)

    return Frob_norm, Frob_norm_X_Xapprx

def hist(Frob_norm, bins):
    '''Plotting histogram'''
    numelems = Frob_norm.shape[0] * Frob_norm.shape[1]
    Fn_vector = np.reshape(Frob_norm, (numelems, 1))
    Fn_vector_nonzero = Fn_vector[Fn_vector > 0.]
    Tot_elements = 36255*55       # Total no.of elements in Xapprx
    RMS = Fn_vector_nonzero/Tot_elements  
    
    mu = np.mean(RMS)
    sigma = np.std(RMS)
     
    num_bins = bins
    n, bins, patches = plt.hist(RMS, num_bins, density = True)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
   
    plt.plot(bins, y, '--', color ='black')
    
    plt.xlabel('RMS')
    plt.ylabel('Frequency')
 
    plt.title('Histogram', fontweight = "bold")
 
    plt.show()

    
# def hist_both(Frob_norm, Frob_norm_X_Xapprx, bins):

#     numelems = Frob_norm.shape[0] * Frob_norm.shape[1]
#     Fn_vector = np.reshape(Frob_norm, (numelems, 1))
#     Fn_vector_nonzero = Fn_vector[Fn_vector > 0.]
#     Tot_elements = 36255*55       # Total no.of elements in Xapprx
    
#     RMS1 = Fn_vector_nonzero/Tot_elements  
#     RMS2 = Frob_norm_X_Xapprx/Tot_elements
    
#     mu1 = np.mean(RMS1)
#     sigma1 = np.std(RMS1)
#     mu2 = np.mean(RMS2)
#     sigma2 = np.std(RMS2)
#     num_bins = bins
#     # num_bins_Xapprx_values = bins_Xapprx_values
#     # num_bins_X_Xapprx_values = bins_X_Xapprx_values
    
#     n, bins, patches = plt.hist(RMS1, num_bins, density = True)
#     y1 = ((1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * (1 / sigma1 * (bins - mu1))**2))
#     plt.plot(bins, y1, '--', color ='blue', label='RMS between Xapprx values')

#     n, bins, patches = plt.hist(RMS2, num_bins, density = True)
#     y2 = ((1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * (1 / sigma2 * (bins - mu2))**2))
#     plt.plot(bins, y2, '--', color ='orange', label='RMS between X and Xapprx')
    
#     plt.xlabel('(RMS)')
#     plt.ylabel('Frequency')
 
#     plt.title('Histogram of RMS', fontweight = "bold")
#     #print("RMS1= ",RMS1, "\nRMS2= ",RMS2)
#     plt.legend()
#     plt.show()

def hist_both(Frob_norm, Frob_norm_X_Xapprx, filename:str):

    numelems = Frob_norm.shape[0] * Frob_norm.shape[1]
    Fn_vector = np.reshape(Frob_norm, (numelems, 1))
    Fn_vector_nonzero = Fn_vector[Fn_vector > 0.]
    Tot_elements = 36255*55       # Total no.of elements in Xapprx
    
    RMS1 = Fn_vector_nonzero/Tot_elements  
    RMS2 = Frob_norm_X_Xapprx/Tot_elements
    
    maxRMS = np.max((np.max(RMS1), np.max(RMS2)))
    minRMS = np.min((np.min(RMS1), np.min(RMS2)))
    
    mu1 = np.mean(RMS1)
    sigma1 = np.std(RMS1)
    mu2 = np.mean(RMS2)
    sigma2 = np.std(RMS2)
    
    num_bins_Xapprx_values = 100
    num_bins_X_Xapprx_values = 100

    fig,ax = plt.subplots(figsize=(6,4));
    
    n, bins, patches = plt.hist(RMS1, num_bins_Xapprx_values, range = (minRMS, maxRMS), density = True, alpha = 0.5)
    y1 = ((1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * (1 / sigma1 * (bins - mu1))**2))
    plt.plot(bins, y1, '--', color ='blue', label='RMS between different approximations')

    n, bins, patches = plt.hist(RMS2, num_bins_X_Xapprx_values, range = (minRMS, maxRMS), density = True, alpha = 0.5)
    y2 = ((1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * (1 / sigma2 * (bins - mu2))**2))
    plt.plot(bins, y2, '--', color ='orange', label='RMS between data and approximations')
    
    plt.xlabel('RMS (x$10^5$)')
    plt.ylabel('Prob. density (x$10^6$)')
    #plt.title('Histogram of RMS', fontweight = "bold")
    #print("RMS1= ",RMS1, "\nRMS2= ",RMS2)
    plt.legend()
    plt.show()
    return(fig)


def local_minimum(k, iter, tol, Xgt, filename:str ):
    "Running NMF for a given number of metagenes with a given tolerance value for given number of iterarions, Computing Frobenius norm among the Xapprx values returned in the function NMF_Results, Plotting histogram for frobenius norm between X and Xapprx "
    
    Xapprx_both = NMF_Results(k, iter,  tol, Xgt)#Returns Xapprx after running NMF for 10 iterations for 10 metagenes
    Xapprx_both_pairwise_Frob = Frob_iter(Xgt, Xapprx_both[0], iter)
    plot_both = hist_both(Xapprx_both_pairwise_Frob[0], Xapprx_both_pairwise_Frob[1], filename)
    
    return plot_both



def sorted_Hmt_behaviors(Hmt, Wgm):
    "Return W matrix(Wgm), normalized H matrix(normalized_Hmt) and indices(kK), which are sorted according to metagene behavior. Metagene with smallest dominant value(over its timepoints) will be the first in the sorted array, while metagene with largest dominant value will be the last. "   
    
    mkDominant = [np.argmax(Hm) for Hm in Hmt] # for each metagene, mkDominant is the value of m such that H[m,t] is greatest
    m_mp = np.argsort (mkDominant)               # m_mp[0] is the m_mp such that mkDominant is smallest (earliest upgrulation). Therfore, m_mp is an array of indices which are sorted according to the ascending order of highest value in H[m,t].
    Hmt = Hmt[m_mp,:]                            # take Hmt but reorder the rows according to permutation m_mp
    Wgm = Wgm[:, m_mp]
    Hmt_size = Hmt.shape
    normalized_Hmt = np.zeros(Hmt_size)
    for m in range(Hmt_size[0]):
        normalized_Hmt[m,:] = Hmt[m,:] / (np.max(Hmt[m,:]))
    
    return Wgm, normalized_Hmt, m_mp



def NMF_sort(X, W, metagene, TotGeneNameList):
    ''' Sort the original dataset matrix X and weight matrix W, in descending order of the weights of a given metagene. Returns sorted X, W matrices and gene names'''
    
    w_meta = W[:,metagene]
    
    # Manu 03/20/2025: Changed to descending order to make it easier to pick out high weight genes
    w_meta_sortindices = np.argsort(w_meta, axis=0)[::-1] # "w_meta_sortindices" is an array of indices that sort w_meta, along axis 0

    # Manu 03/20/2025: w_meta_sorted is not being used for anything
    #w_meta_sorted = np.sort(w_meta, axis=0)         # "w_meta_sorted" is the sorted array along axis 0
    
    X_sorted = X[w_meta_sortindices,:]              # "X_sorted" is sorted X according to the indices of the sorted W matrix
    W_sorted = W[w_meta_sortindices,:]              # "W_sorted" is sorted W according to the indices of the sorted W matrix

    gene_names = TotGeneNameList[w_meta_sortindices] # "gene_names" is sorted gene namess according to the indices of the sorted W matrix
    
    # plt.xlabel('Timepoints')
    # plt.ylabel('Genes')
    # plt.imshow(X_sorted[::-1], aspect='auto',interpolation='nearest')
    
    return X_sorted, W_sorted, gene_names, w_meta_sortindices

def writeGenesWeights(W, geneNames, metagene, howmany, directory):
    ''' Given metagene metagene, weights matrix W sorted in descending order of the weights of metagene, 
        and similarly sorted geneNames, write the top howmany geneNames and W[:metagene] to a CSV file in
        the directory directory '''

    outWmNames = np.concatenate( \
                                ( \
                                 np.reshape(geneNames[:howmany], (-1,1)), \
                                 np.reshape(W[:howmany,metagene], (-1,1)) \
                                ), \
                                axis = 1 \
                               )
    filename = 'metagene_%d_top_%d_genes.csv' % (metagene, howmany)

    with open(os.path.join(directory, filename) , 'w',  newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Gene', 'Wg%d' % metagene])
        writer.writerows(outWmNames) 

# def NMF_Sort_all_metagenes(k, Xgt, Wgm, TotGeneNameList):
#     '''Sorting X matrices and gene names for all the metagenes, doing a for loop over all the metagenes, sorting one metagene at a time. Returns numpy nd arrays of sorted X matrices, sorted gene names  and the sorted W matrices'''
    
#     Xsorted_all_metagenes = np.zeros((k, Xgt.shape))
#     gene_names_all_metagenes = np.zeros((k, Xgt.shape))
#     Wsorted_all_metagenes = np.zeros((k, Xgt.shape))
    
#     for num_metagene in range(k):
#         sorted_metagene = NMF_sort(Xgt, Wgm, num_metagene, TotGeneNameList)
        
#         Xsorted_all_metagenes = sorted_metagene[0]
#         gene_names_all_metagenes = sorted_metagene[1]
#         Wsorted_all_metagenes = sorted_metagene[2]
        
#     return Xsorted_all_metagenes, gene_names_all_metagenes, Wsorted_all_metagenes  
    

def NMF_nested_sort(Xgt, Wgm, Hmt, TotGeneNameList):
    "Nested sorting for NMF 3D visualization. Returns sorted Xgt, Wgm and normalized Hmt"
    
    G,T = Xgt.shape
    G,M = Wgm.shape
    
    # Sort metagenes m so that the earliest upregulation behavior is m=0, etc.
    W_g_mp, H_mp_t, m_mp = sorted_Hmt_behaviors(Hmt,Wgm)

    # Sort g
    mp_g = [np.argmax(row) for row in W_g_mp]        # for each gene g, find the metagene m that dominates it (or else, indices of the dominating metagene m, for each gene)
    W_g = [ + W_g_mp[g,mp_g[g]] for g in range(G)]   # also record the weight of that dominant metagene  (can change + to - if you like)
    g_gp = np.lexsort(np.array([W_g,mp_g]))          # "g_gp" is an array of indices that sort weights of the dominant metagene for all genes.

    X_gp_t = Xgt[g_gp,:]                 # arrange the genes in Xgt in the desired order
    
    W_gp_mp = Wgm[g_gp,:][:,m_mp]        # arrange g and m in Wgm   (this is ok)
    gene_names = [TotGeneNameList[j] for j in g_gp]   # "gene_names" is sorted gene names according to the indices that sort weights of dominant metagene for all genes
    
    return X_gp_t, W_gp_mp, H_mp_t, g_gp, gene_names

def NMF_sorted_visualization(X_gp_t, W_gp_mp, H_mp_t, Timepoints, fname:str):

    "Visualizing NMF as a 3D image. "

    G,T = X_gp_t.shape
    G,M = W_gp_mp.shape
    Timepoints = Timepoints.reshape(X_gp_t[1].shape,)
    Timepoints = Timepoints.flatten()
    #print(Timepoints)
    tmin,tmax = min(Timepoints),max(Timepoints)
    
    affine2D = mpl.transforms.Affine2D
    fig,ax = plt.subplots(figsize=(24,16));
    b = 0*  .2      # this determines how far faces are offset from axis, as fraction of cell
    s=0.001

    u=0   # .1
    v=.5
    w=.7
    q=1 # scaling of right axis (timepoints)
    #======== Draw Xgt on top face
    trans = affine2D([[q,-w,0],[u,v,0],[0,0,1]]) + ax.transData
    im = plt.imshow(X_gp_t,extent=(b,b+T,b,b+s*G),origin='lower',interpolation='nearest')               # NOTE X_gp_t
    im.set_transform(trans)
    x1,x2,y1,y2 = im.get_extent()
    ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], lw=1, c='#999', transform=trans)
    plt.colorbar(shrink = 0.2, label = 'Expression')
    
    #======== Draw Wgm on left face
    trans = affine2D([[0,-w,0],[-1,v,0],[0,0,1]]) + ax.transData
    WgmYAxisAngle = np.rad2deg(np.arctan(-v/w))
    im = plt.imshow(W_gp_mp,extent=(b,b+M,b,b+s*G),interpolation='nearest',origin='lower',vmin=0,vmax=1.5)                # NOTE THIS IS W_gp_m
    im.set_transform(trans)
    x1,x2,y1,y2 = im.get_extent()
    ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], lw=1, c='#999', transform=trans)  # draw bounding parallelogram
    ax.text (x2+1, ((y1+y2)/2)-2, "Gene g       ", ha='right', va='center', rotation = WgmYAxisAngle, transform=trans)
    ax.text ((x1+x2)/2, y2+1, 'Metagene m     ', ha='right', va='center', rotation='vertical', transform=trans)
    for g in range(0,G,10000): ax.text (b+x2+.2, b+s*g+1, f'{g} ', ha='right', va='top', transform=trans)
    for m in range(M): ax.text (b+x1+m, y2, f'{m+1} ', ha='right', va='top', transform=trans)

    #======== Draw Hmt on right face
    trans = affine2D([[q,0,0],[u,-1,0],[0,0,1]]) + ax.transData
    # im = plt.imshow(H_mp_t,extent=(b,T,b,M),interpolation='nearest',origin='lower')
    # im.set_transform(trans)
    #plt.plot(H_mp_t.T*4 + np.arange(10), transform=trans);
    x1,x2,y1,y2 = b,T+b,b,M+b   #im.get_extent()
    ax.fill([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], lw=1, c='#999', fc='k', transform=trans)  # maybe c should be ec
    ax.text ((x1+x2)/2, y2+1, 'Timepoint t',ha='center',  va='top', transform=trans)
    ax.text (x2+1, (y1+y2)/2, '      Metagene m', ha='left', va='center', rotation='vertical', transform=trans)
    for t in range(T): ax.text (b+t, y2, f'{Timepoints[t]:.0f}', ha='center', va='top', transform=trans, rotation = 'vertical')
    for m in range(M): ax.text (b+x2, b+y1+m, m+1, ha='left', va='top', transform=trans)
    for m in range(M):
        yoffset = (m+1) *1.0   # note: start from bottom of figure
        yscale = -.9   # note negative sign
        y = H_mp_t[m,:]*yscale + yoffset
        plt.plot(y, 'w', lw=.5, transform=trans)
        
        plt.fill_between(range(T), y, yoffset, fc='yellow', transform=trans)
    #plt.xlim();plt.ylim(); ax.set_aspect('equal');


    #======== Face labels
    ax.text (-14,.5*s*G+.05*T, r"Gene expression $X_{gt}$", size=16, c='#999')
    ax.text (-w*s*G+4,-3, r"Weights $W_{gm}$", size=16, rotation = WgmYAxisAngle, c='#999')
    ax.text (14,-13, r'Behaviors $H_{mt}$', size=16, c='#999')

#    ax.set_xlim(-30, 60)  # hardwired.  would needto compute this by hand
#    ax.set_ylim(-20,23)
    #ax.autoscale()
    #ax.use_sticky_edges = False
    #ax.margins(0.05, 0.05)
    ax.axis ('off');
    #fig.savefig(fname, bbox_inches='tight')
    return(fig)

def behavior_visualization(X_gp_t, W_gp_mp, H_mp_t, Timepoints):

    "Visualizing behaviors. "

    G,T = X_gp_t.shape
    G,M = W_gp_mp.shape
    Timepoints = Timepoints.reshape(X_gp_t[1].shape,)
    Timepoints = Timepoints.flatten()
    #print(T,M)
    tmin,tmax = min(Timepoints),max(Timepoints)
    
    fig,ax = plt.subplots(figsize=(12,8));
    b = 0*  .2      # this determines how far faces are offset from axis, as fraction of cell
    s=0.001

    u=0   # .1
    v=.5
    w=.7
    q=1 # scaling of right axis (timepoints)
    

    #======== Draw Hmt 
    x1,x2,y1,y2 = b,T+b,b,M+b   #im.get_extent()
    ax.fill([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], lw=1, c='#999', fc='k')  # maybe c should be ec
    ax.text ((x1+x2)/2, y1-1, 'Timepoint t',ha='center',  va='top')
    ax.text (x1-2, (y1+y2)/2, '      Metagene m', ha='left', va='center', rotation = 'vertical')
    for t in range(T): ax.text (b+t, y1-0.2, f'{Timepoints[t]:.0f}', ha='center', va='top', rotation = 45)
    for m in range(M): ax.text (x1-0.5, 0.2+y1+m, m, ha='left', va='top')
    for m in range(M):
        yoffset = (m+0) *1.0   # note: start from bottom of figure
        yscale = .9   # note negative sign
        y = H_mp_t[m,:]*yscale + yoffset
        plt.plot(y, 'w', lw=.5)
        plt.fill_between(range(T), y, yoffset, fc='yellow')
    #plt.xlim();plt.ylim(); ax.set_aspect('equal');

    ax.text (7,-8, r'Behaviors $H_{mt}$', size=16, c='#999')
    ax.axis ('off');


def FrobNorm_Metagene_comparator(X, tol, metagene_list):
    '''Computing change in RMSE for approximate X matrices for consecutive number of metagenes'''
    
    Frob_norm_list = []
    prevX = X
    for metagene in metagene_list:
        w, h, Frob_norm = NMF(X, metagene, tol)
        #Frob_norm_list.append(Frob_norm[-1])
        nextX = w.dot(h)
        Frob_norm_list.append(la.norm(prevX - nextX))
        prevX = nextX
    
    #print(Frob_norm_list)
        
    return Frob_norm_list

def FrobNorm_Metagene_comparator_new(X, tol, metagene_list):
    '''Computing RMSE of approximate X matrices and X original matrix for consecutive number of metagenes'''
    
    Frob_norm_list = []
   
    for metagene in metagene_list:
        w, h, Frob_norm = NMF(X, metagene, tol)
        approx_X = w.dot(h)
        Frob_norm_list.append(la.norm(approx_X-X))
        
    return Frob_norm_list

def FrobNorm_Metagene_comparator_both(X, tol, metagene_list):
    '''Both RMSE plots for consecutive number of metagenes'''
    
    Frob_norm_list_change_RMSE = []
    Frob_norm_list_RMSE = []
    prevX = X
    for metagene in metagene_list:
        w, h, Frob_norm = NMF(X, metagene, tol)
        #Frob_norm_list.append(Frob_norm[-1])
        approx_X = w.dot(h)
        nextX = w.dot(h)
        Frob_norm_list_change_RMSE.append(la.norm(prevX - nextX))
        prevX = nextX
        Frob_norm_list_RMSE.append(la.norm(approx_X-X))
    
    #print(Frob_norm_list)
    
    return Frob_norm_list_change_RMSE, Frob_norm_list_RMSE

def Tusi_behavior_plots(behaviorExpr, cols, rows, tusi_orig_coord):
    ''' This function plots scatter plots for given number of behaviors for Tusi data. 
Inputs of the function are behavior matrix  for given number of behaviors (ex. behaviorExpr_unscaled_40 matrix is 4763*40 by size, and NMF was run on unscaled X matrix of gene expressions), no.of columns and no.of rows as integers for the subplots, and original tusi coordinates array. Setting color range is based on log scale'''
   
    num_behaviors = behaviorExpr.shape[1]

    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axs = axs.flatten()  

    x_axis = tusi_orig_coord[:,1]
    y_axis = tusi_orig_coord[:,2]

# set the same color limits for all plots
    vmin = np.min(behaviorExpr[behaviorExpr > 0])# avoid log(0)
    vmax = np.max(behaviorExpr)
    

    for i in range(num_behaviors):
        sc = axs[i].scatter(
            x_axis, y_axis, marker=".", c=behaviorExpr[:,i],
            cmap="viridis", s=5, norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        axs[i].set_title(f"Behavior {i}")
        axs[i].axis("equal")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
# Add a single colorbar on the right for all subplots
    cbar = fig.colorbar(sc, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Expression")

# Shared axis labels
    # fig.supxlabel("x_coordinate")
    # fig.supylabel("y_coordinate")
    #plt.tight_layout(rect = [0,0,0.85,1])
    fig.subplots_adjust(right = 0.85)

    plt.show()

def Tusi_single_behavior_plot(behaviorExpr, tusi_orig_coord, which_behavior):
    '''This function plots scatter plot the for given number of behavior for Tusi data. 
Inputs of the function are behavior matrix for given behavior (ex. behaviorExpr matrix is 4763*40 by size), original tusi coordinates array, and which behavior you need to plot.Setting color range is based on log scale'''
    x_axis = tusi_orig_coord[:,1] 
    y_axis = tusi_orig_coord[:,2] 
    values = behaviorExpr[:,which_behavior] 

    positive_values = values[values > 0]
    vmin = np.min(positive_values)
    vmax = np.max(values)
    #vmax = np.percentile(values, 97)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        x_axis, y_axis,
        marker=".", c=values,
        cmap="viridis", s=5,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    
    plt.colorbar(label="behavior") 
    # plt.xlabel("x_coordinate") 
    # plt.ylabel("y_coordinate") 
    plt.title("Scatter plot of Tusi data for behavior " + str( which_behavior)) 
    plt.axis("equal") 
    plt.xticks([])
    plt.yticks([])
    
    plt.show()

# Manu 10/01/2025: function to plot multiple behaviors using Tusi et al's published
# SPRING coordinates. Uses plotTusiBehavior() so that the expression data is sorted 
# to ensure that low-expression cells don't hide high expression cells. Also the mapping 
# is chosen so that the 99.9 %ile is maximum so it is easy to see areas of peak expression.
def plotMultipleTusiBehaviors(behaviorExpr, cellCoords, cols, rows, percentile = 99.9):
    
    numBehaviors = behaviorExpr.shape[1]

    # create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten()  

    for i in range(numBehaviors):
        sc = plotTusiBehavior(behaviorExpr[:,i], cellCoords, axis = axes[i], percentile = percentile)
        axes[i].set_title(f"Behavior {i}")
    for j in range(numBehaviors, len(axes)):
        axes[j].axis('off')    
    
    # Add a single colorbar on the right for all subplots
    cbar = fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Expression")
    
    plt.show()
    return fig

    
# Manu 10/01/2025: function to plot behavior expression using Tusi et al's published
# SPRING coordinates. The expression data is sortedto ensure that low-expression cells
# don't hide high expression cells. Also the mapping is chosen so that the 95 %ile
# is maximum so it is easy to see areas of peak expression.
def plotTusiBehavior(behaviorExpr, cellCoords, axis = None, percentile = 99.9):

    annotate = False

    # Sort the behavior in order of increasing expression and the coordinates in 
    # the same order
    sortIndices = np.argsort(behaviorExpr)
    sortedBehav = behaviorExpr[sortIndices]
    sortedCoords = cellCoords[sortIndices,:]
   

    # If being run alone on one behavior then create the figure
    if axis is None:
        annotate = True
        fig, axis = plt.subplots() 
        
    sc = axis.scatter(
        sortedCoords[:,1], 
        sortedCoords[:,2], 
        marker=".", 
        c=sortedBehav,
        cmap="viridis", 
        s=5,
        # Set the max of the color scale to the specified percentile (99.9 default)
        # so that a minimum of cells (0.1%) have max brightness and that it is easy to see 
        # the patterns
        vmax = np.percentile(sortedBehav, percentile)
    )
    
    #print(sortedBehav.shape, sortedCoords.shape)
    # make sure the points are circles
    axis.set_aspect('equal', adjustable='datalim')
    axis.set_xticks([])
    axis.set_yticks([])
    
   
    # If being run alone on one behavior then create colorbar and show it
    if annotate:
        fig.colorbar(sc, ax=axis, label = "Expression")
        fig.show()

    return sc
