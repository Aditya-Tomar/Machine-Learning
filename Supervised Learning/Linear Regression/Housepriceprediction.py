import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def featureScaling(X,Y):
    for i in range(0,X.shape[1]):
        mean = (np.sum(X[:,i])/Y.size)
        X[:,i] = np.around((X[:,i] - mean)/np.max(X[:,i]),5)
    Y = (Y - (np.sum(Y)/Y.size))/np.max(Y)
    return X,Y
    
def hypothesis(X,theta):
    return X*(theta.transpose())

def costFunction(X,Y,theta):

    squarred_error = np.multiply((X*theta.transpose() - Y),(X*theta.transpose() - Y))
    cost = np.sum(squarred_error)/(2*Y.size)

    return cost


def gradientDescent(X,Y,theta,alpha):
    j=0
    l=[]
    val=50
    print(theta)
    Theta1 = theta.transpose()
    while(j<val):
        #Theta1 = np.matrix(np.zeros((theta.size,1),dtype = float))
        for i in range(0,theta.size):
            Theta1[i,0] = theta.transpose()[i,0] - alpha * ((np.sum((X[:,i].transpose()*(X*theta.transpose() - Y))))/Y.size)
            
        theta = Theta1.transpose()               
        l.append(costFunction(X,Y,Theta1.transpose()))
        j+=1
    #print("Theta",Theta1)
    #print(theta)
    #print(l)
    plt.plot(range(0,val),l[:])
    plt.title("Cost value evaluation")
    plt.xlabel("Iterations")
    plt.ylabel("Cost value")
    plt.show()
    return theta
    #return Theta1.transpose()


if __name__ == "__main__":
    
    df = pd.read_csv(r'E:\Adi Doc\Python\Linear regression\Housepriceprediction\Dataset\housepricing.zip',dtype = float)

    #Training Data 70%
    #Selecting features 

    X = df[['Area','Garage','FirePlace','White Marble','Black Marble','Floors','Solar','Electric','Glass Doors','Garden']][:int(df.shape[0]*.7)].to_numpy()
    Y = df[['Prices']][:int(df.shape[0]*.7)].to_numpy()

    X = df[['Area']][:int(df.shape[0]*.7)].to_numpy()
    Y = df[['Prices']][:int(df.shape[0]*.7)].to_numpy()

    #feature scaling
    X,Y = featureScaling(X,Y)

    #converting numpy to matrix
    X = np.matrix(np.around(X,5))
    Y = np.matrix(np.around(Y,5))

    # Adding extra feature 1 as bias term to X
    bias = np.ones((int(df.shape[0]*.7),1),dtype=np.int8)
    X = np.matrix(np.c_[bias,X])

    #Theta Coefficient initialization
    #theta = np.around((np.random.rand(X.shape[1])),5)
    theta = np.array([0.49849,0.20521])
    theta = np.matrix(theta)

    #defining learning rate
    alpha = 0.1

    #Taking output depend on random theta value
    line = hypothesis(X,theta)

    #print(line)
    
    plt.plot(X[:200,1],Y[:200,0],"o",X[:200,1],line[:200,:])
    plt.title("Untrained Model")
    plt.xlabel("X - value")
    plt.ylabel("Y - value")
    plt.show()

    #Learning the Theta value or minimizing the cost function
    theta = gradientDescent(X,Y,theta,alpha)

    #Fitting the line on data for value prediction
    line = hypothesis(X,theta)[:200]

    plt.plot(X[:200,1],Y[:200,0],"o",X[:200,1],line)
    plt.title("Trainned Model")
    plt.xlabel("X - label")
    plt.ylabel("Y - label")
    plt.show()

    ##print(X)
    ##line = hypothesis(X,theta)
    ##plt.plot(X[:2000,1],Y[:2000,0],"o",X[:2000,1],line[:2000])
    ##plt.show()


