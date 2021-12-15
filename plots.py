import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

def to_image(vector):
    '''Outputs vector of size n^2 into a nxn matrix form'''
    n = np.sqrt(len(vector))
    n = np.int(n)
    final = []
    for i in range(n):
        loc = i*n
        final.append(vector[loc:loc+n])
    return np.array(final)

def rand_samples(X_data, y_data, letters = False):
    '''Plots 3x3 grid with random images from X_data with their labels
    If X_data contains samples of letters then use letters=True
    '''
    num_samp = X_data.shape[0]
    samp = np.random.randint(1,num_samp, 9)
    fig, ax = plt.subplots(3,3,figsize = (15, 15))

    for i in range(3):
        for j in range(3):
            k = i*3+j
            fig = ax[i,j].figure
            image = to_image(X_data[samp[k]])
            if letters == False:
                ax[i,j].set_title('True label: '+str(int(y_data[samp[k]])))
            elif letters == True:
                image = np.rot90(np.fliplr(image),1)
                ax[i,j].set_title('True label: '+chr(ord('`')+int(y_data[samp[k]])))
            ax[i,j].imshow(image, cmap='gray')
            ax[i,j].set(xticklabels=[],yticklabels=[])

    return fig

def predict_set(p_values, sig_level):
    '''Returns each prediction set for each sample for a defined significance level'''
    final = []
    for sample in p_values:
        pred_set = np.where(sample>=sig_level) # find where each sample has a p-value greater than epsilon for each label
        final.append(list(pred_set[0]))
    return final

def calibration_curve(y_test, p_values, sig_levels = np.arange(0,1,0.001),):
    ''' Plots a calibation and performance curve'''
    error_rates = np.zeros(len(sig_levels)) # to store error rates
    mult_rates = np.zeros(len(sig_levels))
    for i, sig in enumerate(sig_levels):
        temp_set = predict_set(p_values,sig)
        for loc, predictions in enumerate(temp_set):
            if y_test[loc] not in predictions:
                error_rates[i] += 1
            if len(predictions)>1:
                mult_rates[i] += 1
    error_rates = error_rates/len(y_test)
    mult_rates = mult_rates/len(y_test)
    fig = plt.figure()
    plt.plot(sig_levels,error_rates, linestyle='dashed', label = 'Calibration curve')
    plt.plot(sig_levels,mult_rates, label = 'Performace curve')
    plt.plot(sig_levels,sig_levels, color = 'k', label = 'x=y')
    plt.legend()
    plt.xlabel('Significance level')
    plt.ylabel('Percentage of errors and multiple predictions')
    plt.show()
    return fig

def corr_incorr_plot(p_values, y_test):
    '''Plots confidence vs credibility with correct and incorrect predictions'''
    predictions = np.argmax(p_values,axis=1)
    correct = p_values[predictions==y_test] # get p-values of correct predictions
    correct_conf_cred = np.partition(correct,-2, axis=1)
    wrong = p_values[predictions!=y_test] # get p-values of incorrect predictions
    wrong_conf_cred = np.partition(wrong,-2, axis=1)
    fig = plt.figure()
    plt.scatter(1 - correct_conf_cred[:,-2], correct_conf_cred[:,-1], color='g', alpha = 0.2, s=10)
    plt.scatter(1 - wrong_conf_cred[:,-2], wrong_conf_cred[:,-1], color='r', alpha=0.2, s=10)
    plt.legend(['Correct','Incorrect'])
    plt.title('Confidence vs credibility of correct and incorrect predictions')
    plt.xlabel('Confidence')
    plt.ylabel('Credibility')
    plt.show()
    return fig

### For OCP

def cumm_plot(p_values, y_data, sig_level):
    '''Plots the cummulative errors, empty and multple predictions for an OCP'''
    rows = len(y_data)
    err = 0
    mult = 0
    emp = 0
    err_store = np.zeros(rows)
    mult_store = np.zeros(rows)
    emp_store = np.zeros(rows)
    for i, sample in enumerate(p_values):
        predict_set = np.where(sample>=sig_level)[0]
        if y_data[i] not in predict_set:
            err += 1
            err_store[i] = err
        elif y_data[i] in predict_set:
            err_store[i] = err
        if predict_set.size==0:
            emp += 1
            emp_store[i] = emp
        if predict_set.size!=0:
            emp_store[i] = emp
        if predict_set.size>1:
            mult += 1
            mult_store[i] = mult
        if predict_set.size<=1:
            mult_store[i] = mult
    x = np.arange(rows)
    fig = plt.figure()
    plt.plot(x, err_store, label='Errors')
    plt.plot(x, emp_store, label='Empty predictions')
    plt.plot(x, mult_store, label='Multiple predictions')
    plt.xlabel('Examples')
    plt.ylabel('Cumulative errors, multiple and empty predictions')
    plt.legend(loc=2)
    plt.show()
    return fig

def err_plot(p_values,y_data, sig_levels = np.array([0.01, 0.05, 0.1, 0.2])):
    ''' Plots the cummulative errors for different significance levels for an OCP'''
    rows = len(y_data)
    x = np.arange(rows)
    fig = plt.figure()
    for sig_level in sig_levels:
        err = 0
        mult = 0
        emp = 0
        err_store = np.zeros(rows)
        mult_store = np.zeros(rows)
        emp_store = np.zeros(rows)
        for i, sample in enumerate(p_values):
            predict_set = np.where(sample>=sig_level)[0]
            if y_data[i] not in predict_set:
                err += 1
                err_store[i] = err
            elif y_data[i] in predict_set:
                err_store[i] = err
        plt.plot(x, err_store, label=str(sig_level*100)+'%')
    plt.xlabel('Examples')
    plt.ylabel('Cumulative errors at different confidence levels')
    plt.legend(title = 'Significance level')
    plt.show()
    return fig

def ave_false_p(p_values, y_data):
    '''Plots the average false p-value for each epoch for an OCP'''
    rows = len(y_data)
    x = np.arange(rows)
    indices = np.arange(len(np.unique(y_data)))
    values = np.zeros(rows)

    for i, p in enumerate(p_values):

        predict_set = p[indices!=y_data[i]]
        ave_false = np.mean(predict_set)
        values[i] = ave_false


    fig = plt.figure(figsize=(16,8))
    plt.plot(x, values)

    plt.xlabel('Examples')
    plt.ylabel('Average false p-value')
    plt.show()

    return fig

def ave_set_plot(p_values, epsilons = np.array([0.01, 0.05, 0.1, 0.2])):
    '''Plots the average prediction set size after each epoch for an OCP'''
    rows = p_values.shape[0]
    averages = np.zeros(rows) # to store average set size for each epsilons
    fig = plt.figure()
    for i, eps in enumerate(epsilons):
        for j in range(rows):
            temp = np.sum(p_values[:j+1]>eps,axis=1) # first average of set size
            averages[j] = np.average(temp) # then overall average
        plt.plot(np.arange(rows),averages, label = str(eps*100)+'%')
    plt.xlabel('Samples trained on')
    plt.ylabel('Average prediction set size')
    plt.ylim([0,10])
    plt.legend(title='Significance level')

    return fig

def cred_samples(conf_table, X_test, min_cred=0.99, max_cred=1, heading=True):
    '''Plots 3x3 images of predictions with a defined range of credibility'''
    cred_table = conf_table[:,-1]
    cred_great = np.where(cred_table>min_cred) # locations of samples that have very high credibility
    cred_low =  np.where(cred_table<=max_cred)
    cred_loc = np.intersect1d(cred_great,cred_low)
    cred_values = conf_table[cred_loc]
    cred = X_test[cred_loc]
    cred_df = pd.DataFrame(cred_values, columns=['Label', 'Prediction', 'Confidence', 'Credibility'])
    samp_values = cred_df.sample(9)
    samp_idx = samp_values.index
    samp = X_test[cred_loc[samp_idx]]
    fig, ax = plt.subplots(3,3,figsize = (15, 15))
    loc = cred_loc[samp_idx]
    table = conf_table[loc]
    for i in range(3):
        for j in range(3):
            k = i*3+j
            fig = ax[i,j].figure
            if heading==True:
                ax[i,j].set_title('True label: '+str(int(table[k,0]))+'   Predicted Label: '+str(int(table[k,1]))+
                '\nConfidence: '+str(round(table[k,2],4))+'   Credibility: '+str(round(table[k,3],4)))
            else:
                ax[i,j].set_title('True label: '+str(int(table[k,0]))+'   Predicted Label: '+str(int(table[k,1])))
            ax[i,j].imshow(to_image(samp[k]), cmap='gray')
            ax[i,j].set(xticklabels=[],yticklabels=[])
    return fig

### For anomaly detection

def anomaly_eps(conf_table):
    '''Plots the number of the anomalies detected for different anomaly thresholds'''
    sig_levels = np.arange(0,1,0.001)
    num_anom = np.zeros(len(sig_levels))
    fig = plt.figure()
    for i, sig in enumerate(sig_levels):
        num_anom[i] += np.sum(conf_table[:,-1]<sig)
    plt.plot(sig_levels,num_anom)
    plt.xlabel('Anomaly threshold $\epsilon$')
    plt.ylabel('Anomalies detected')
    return fig

def anomaly_eps_errs(p_values):
    '''Plots the frequency of the anomalies detected for different anomaly thresholds'''
    rows = p_values.shape[0]
    sig_levels = np.arange(0,1,0.001)
    num_anom = np.zeros(len(sig_levels))
    fig = plt.figure()
    for i, sig in enumerate(sig_levels):
        temp = p_values<sig
        num_anom[i] = np.sum(np.all(temp, axis=1))
    num_anom = num_anom/rows
    plt.plot(sig_levels,num_anom)
    plt.xlabel('Anomaly threshold $\epsilon$')
    plt.ylabel('Percentage of anomlies detected')

    return fig

def anomaly_online(p_values, sig_levels = np.array([0.01, 0.05, 0.1, 0.2])):
    '''Returns plot of number of anomalies detected for different anomaly threshold for each epoch'''
    rows = p_values.shape[0]
    fig = plt.figure()
    for sig in sig_levels:
        anom = 0
        num_anom = np.zeros(rows)
        for i, sample in enumerate(p_values):
            temp = np.any(sample>sig)
            if temp==False:
                anom += 1
            num_anom[i] = anom
        plt.plot(np.arange(rows), num_anom, label=str(sig*100)+'%')
    plt.xlabel('Examples')
    plt.ylabel('Anomalies detected')
    plt.legend(title='Anomaly threshold')
    return fig

def anomaly_eps_models(p_values_set, models):

    fig = plt.figure()
    sig_levels = np.arange(0,1,0.001)
    for j, p_values in enumerate(p_values_set):
        rows = p_values.shape[0]
        num_anom = np.zeros(len(sig_levels))
        for i, sig in enumerate(sig_levels):
            temp = p_values<sig
            num_anom[i] = np.sum(np.all(temp, axis=1))
        num_anom = num_anom/rows
        plt.plot(sig_levels,num_anom, label = models[j])
    plt.xlabel('Anomaly threshold $\epsilon$')
    plt.ylabel('Percentage of anomalies detected')
    plt.legend()
    return fig

def anom_detect_curve(p_values_set, models, sig_levels = np.arange(0,1,0.001)):
    '''Plots the frequency of anomalies detected for different anomaly thresholds'''
    fig = plt.figure()
    for j, model in enumerate(p_values_set):
        p_values = p_values_set[model]
        rows = p_values.shape[0]
        num_anom = np.zeros(len(sig_levels))
        for i, sig in enumerate(sig_levels):
            temp = p_values<sig
            num_anom[i] = np.sum(np.all(temp, axis=1))
        num_anom = num_anom/rows
        plt.plot(sig_levels,num_anom, label = models[j])
    plt.xlabel('Anomaly threshold $\epsilon$')
    plt.ylabel('Percentage of anomalies detected')
    plt.legend()
    return fig

def anom_detect_hist(model_pvalues, model_names, hist=20):
    '''Plots a histogram of the credibilities of samples for different models'''
    temp_cred = []
    for model in model_pvalues:
        p_values = model_pvalues[model]
        cred_vals = np.max(p_values, axis=1)
        temp_cred.append(cred_vals)
    temp_cred = np.array(temp_cred).T
    fig = plt.hist(temp_cred, histtype='step', label = model_names, bins = hist)
    plt.xlabel('Credibility')
    plt.ylabel('Anomalies')
    plt.legend()
    return fig

def rand_anom_samples(X_data):
    '''Plots 9 random samples'''
    num_samp = X_data.shape[0]
    samp = np.random.randint(1,num_samp, 9)
    fig, ax = plt.subplots(3,3,figsize = (15, 15))
    for i in range(3):
        for j in range(3):
            k = i*3+j
            fig = ax[i,j].figure
            image = to_image(X_data[samp[k]])
            ax[i,j].imshow(image, cmap='gray')
            ax[i,j
              ].set(xticklabels=[],yticklabels=[])
    return fig
