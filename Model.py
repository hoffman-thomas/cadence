import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pydotplus

def round_elems(inputList, place = '0'):
    degree = '%.'+place+'f'
    return [degree % elem for elem in inputList]
    
def average(lst): 
    return sum(lst) / len(lst)

#Read in Lydia's Excel file 
xl = pd.ExcelFile('Collected Product Data\Data_Observation_10042018.xlsx')

#Convert Raw Observation and Bill of Materials sheets to dataframes
obs_df = xl.parse("RawObs")    
bom_df = xl.parse("BillofMat")

#Shuffle rows
obs_df = obs_df.sample(frac=1).reset_index(drop=True)

#Select Wash observations
obs_df = obs_df.loc[obs_df['Cell'] == 'Wash']

#Define target, extract time column
target_df = obs_df[['Time (min)']]

#Save off data before One-Hot encoding to look at it later
X_test_orig_df = obs_df[19:24]

#Remove columns 
obs_df = obs_df.drop(columns = ['Time (min)', 'SA', 'FA/Part-ID'])

#One Hot encode categorical features
obs_df = pd.get_dummies(obs_df)

#Split Training/Testing Data
X = obs_df[0:18]
y = target_df[0:18]
X_test = obs_df[19:24]

#Multiple models. Model 1 (regr_1) can be tuned for each parameter. 
model_1 = DecisionTreeRegressor(criterion = 'mse', 
                                splitter = 'best', 
                                max_depth = 5, 
                                min_samples_split = 2, 
                                min_samples_leaf = 1, 
                                min_weight_fraction_leaf = 0.0, 
                                max_features = None, 
                                random_state = None, 
                                max_leaf_nodes = None, 
                                min_impurity_decrease = 0.0, 
                                min_impurity_split = None, 
                                presort = False)

model_2 = DecisionTreeRegressor(max_depth = None)

#Fit
model_1.fit(X, y)
model_2.fit(X, y)

#Predict
y_1 = model_1.predict(X_test)
y_2 = model_2.predict(X_test)

#Formatting
orig_times = list(X_test_orig_df['Time (min)'])
headers = list(obs_df)

#Analyze model 1 results
print(" Orig: ", round_elems(orig_times))
print("Est 1: ", round_elems(y_1))
print("Dif 1: ", round_elems(abs(y_1 - orig_times)))
print("MAE 1: ", round(average(abs(y_1 - orig_times)),2))

#Analyze model 2 results
print("\n Orig: ", round_elems(orig_times))
print("Est 2: ", round_elems(y_2))
print("Dif 2: ", round_elems(abs(y_2 - orig_times)))
print("MAE 2: ", round(average(abs(y_2 - orig_times)),2))

#print("\nFeature Importance Model 2")
#for i in range(0,len(obs_df.columns)):
#    if regr_2.feature_importances_[i] > 0.001:        
#        print(round(regr_2.feature_importances_[i],3), headers[i])


dot_data = tree.export_graphviz(model_2, out_file=None, feature_names = list(X) , class_names =list(y)  ) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("tree.pdf") 
graph.write_png("tree.png") 
