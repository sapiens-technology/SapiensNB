SapiensNB (Naive Bayes) is a classification algorithm that returns a probabilistic result based on Bayes Theorem.

# SapiensNB

The SapiensNB or Sapiens for Naive Bayes is a Machine Learning algorithm focused on probabilistic data classification, where the answer for each input is calculated based on the highest probability of similarity between the prediction input and the training inputs. The probabilistic calculation is based on the following mathematical theorem: P(A/B) = P(B/A) x P(A) / P(B), where P is the probability, A is the class and B are the attributes. This theorem can be applied to both numerical classification and textual classification of data.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install SapiensNB.

```bash
pip install sapiensnb
```

## Usage
Basic usage example:
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# model training for learning assimilation
inputs = [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90] # input examples
outputs = ['unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten'] # output examples
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [15, 1, 25, 2, 35, 3, 45, 4, 55, 5, 65, 6, 75, 7, 85, 8, 95, 9] # inputs to be predicted
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
for result in results: print(result) # displays predicted results
# for each entry we will have a dictionary as a response with a key for each possible label containing the probabilistic values of each of them and a final key with the classification of the most likely label
```
```bash
{'unit': 0.16666666666666669, 'ten': 0.6666666666666666, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.09999999999999998, 'ten': 0.8, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.07142857142857145, 'ten': 0.8571428571428571, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.05555555555555558, 'ten': 0.8888888888888888, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.04545454545454547, 'ten': 0.9090909090909091, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.038461538461538436, 'ten': 0.9230769230769231, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.033333333333333326, 'ten': 0.9333333333333333, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.02941176470588236, 'ten': 0.9411764705882353, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
{'unit': 0.026315789473684237, 'ten': 0.9473684210526315, 'classify': 'ten'}
{'unit': 1.0, 'ten': 0.0, 'classify': 'unit'}
```
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# model training for learning assimilation
inputs = [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90] # input examples
outputs = ['unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten'] # output examples
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [15, 1, 25, 2, 35, 3, 45, 4, 55, 5, 65, 6, 75, 7, 85, 8, 95, 9] # inputs to be predicted
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
print([result['classify'] for result in results]) # displays predicted results
# to display only the value of the classification label, simply reference the “classify” key in each of the response dictionaries
```
```bash
['ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit', 'ten', 'unit']
```
You can also define a training matrix, where each vector in the matrix will represent an input.
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# model training for learning assimilation
inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [70, 80]] # two-dimensional matrix with the input examples
outputs = ['units', 'tens', 'units', 'tens', 'units', 'tens', 'units', 'tens'] # output examples
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[2, 3], [20, 30], [4, 5], [40, 50], [6, 7], [60, 70], [8, 9], [80, 90]] # the inputs to be predicted must have the same dimensionality as the learning inputs
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
for result in results: print(result) # displays predicted results
```
```bash
{'units': 1.0, 'tens': 0.0, 'classify': 'units'}
{'units': 0.0, 'tens': 1.0, 'classify': 'tens'}
{'units': 1.0, 'tens': 0.0, 'classify': 'units'}
{'units': 0.0, 'tens': 1.0, 'classify': 'tens'}
{'units': 1.0, 'tens': 0.0, 'classify': 'units'}
{'units': 0.0, 'tens': 1.0, 'classify': 'tens'}
{'units': 1.0, 'tens': 0.0, 'classify': 'units'}
{'units': 0.0, 'tens': 1.0, 'classify': 'tens'}
```
The input parameter only accepts vectors or matrices, but the output parameter accepts arrays with any type of dimensionality.
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# model training for learning assimilation
inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [70, 80]] # two-dimensional matrix with the input examples
outputs = [['units'], ['tens'], ['units'], ['tens'], ['units'], ['tens'], ['units'], ['tens']] # the output elements can be either scalar values or arrays of any dimensionality.
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[2, 3], [20, 30], [4, 5], [40, 50], [6, 7], [60, 70], [8, 9], [80, 90]] # the inputs to be predicted must have the same dimensionality as the learning inputs
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
for result in results: print(result) # displays predicted results
# as this is a response in dictionary format, non-scalar values will have keys in str format
# note that the classification value remains with the same type
```
```bash
{"['units']": 1.0, "['tens']": 0.0, 'classify': ['units']}
{"['units']": 0.0, "['tens']": 1.0, 'classify': ['tens']}
{"['units']": 1.0, "['tens']": 0.0, 'classify': ['units']}
{"['units']": 0.0, "['tens']": 1.0, 'classify': ['tens']}
{"['units']": 1.0, "['tens']": 0.0, 'classify': ['units']}
{"['units']": 0.0, "['tens']": 1.0, 'classify': ['tens']}
{"['units']": 1.0, "['tens']": 0.0, 'classify': ['units']}
{"['units']": 0.0, "['tens']": 1.0, 'classify': ['tens']}
```
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# model training for learning assimilation
inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [70, 80]] # two-dimensional matrix with the input examples
outputs = [['units', 1], ['tens', 10], ['units', 1], ['tens', 10], ['units', 1], ['tens', 10], ['units', 1], ['tens', 10]] # the output elements can be either scalar values or arrays of any dimensionality.
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[2, 3], [20, 30], [4, 5], [40, 50], [6, 7], [60, 70], [8, 9], [80, 90]] # the inputs to be predicted must have the same dimensionality as the learning inputs
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
for result in results: print(result) # displays predicted results
```
```bash
{"['units', 1]": 1.0, "['tens', 10]": 0.0, 'classify': ['units', 1]}
{"['units', 1]": 0.0, "['tens', 10]": 1.0, 'classify': ['tens', 10]}
{"['units', 1]": 1.0, "['tens', 10]": 0.0, 'classify': ['units', 1]}
{"['units', 1]": 0.0, "['tens', 10]": 1.0, 'classify': ['tens', 10]}
{"['units', 1]": 1.0, "['tens', 10]": 0.0, 'classify': ['units', 1]}
{"['units', 1]": 0.0, "['tens', 10]": 1.0, 'classify': ['tens', 10]}
{"['units', 1]": 1.0, "['tens', 10]": 0.0, 'classify': ['units', 1]}
{"['units', 1]": 0.0, "['tens', 10]": 1.0, 'classify': ['tens', 10]}
```
You can use scalar values of any type to compose the input and output elements.
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# model training for learning assimilation
inputs = [[-1, 0, 2.5, 'a', 5j, False], [-10, 20, 37.2, 'b', 12j, True], [-2, 7, 0.1, 'c', -7j, False], [-24, 18, 51.9, 'd', 14j, True]] # example with elements of multiple types
outputs = [0, 1, 0, 1] # classification/labeling of input examples
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
# model execution/prediction phase
inputs = [[-15, 43, 71.4, 'b', 13j, True], [-2, 0, 1.3, 'a', 8j, False], [-54, 19, 67.8, 'd', 22j, True], [-3, 9, 0.3, 'c', -7j, False]] # prediction lists must have the same number of elements as training lists
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
for result in results: print(result) # displays predicted results
```
```bash
{0: 0.0, 1: 1.0, 'classify': 1}
{0: 1.0, 1: 0.0, 'classify': 0}
{0: 0.0, 1: 1.0, 'classify': 1}
{0: 1.0, 1: 0.0, 'classify': 0}
```
To save a pre-trained model you must call the "saveModel" method. This method receives in the parameter called "path" a string with the path and name of the file to be saved. If no value is assigned to the "path" parameter, the file will be saved with a default name.
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# training phase (pattern recognition)
inputs = [[1, 2, 3], [10, 20, 30], [100, 200, 300], [4, 5, 6], [40, 50, 60], [400, 500, 600]] # input examples
outputs = ['units', 'tens', 'hundreds', 'units', 'tens', 'hundreds'] # output examples
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
sapiensnb.saveModel(path='my_model') # saves the pre-trained model with the prefix "my_model"
# in this case the file will be saved in the current directory because no path was defined before the name
# the file will be saved with the name "my_model-136P.nb", that is, the prefix "my_model" defined in "path", plus the number of model parameters followed by the ".nb" extension
# you can rename the saved file to any name you prefer as long as the ".nb" extension is maintained
```
To load a pre-trained model without the need to train it again, simply call the "loadModel" method, which will receive the address and file name of the model to be loaded in the "path" parameter.
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# prediction phase or execution phase (application of knowledge)
sapiensnb.loadModel(path='my_model-136P.nb') # loads the pre-trained model
inputs = [[2, 3, 4], [20, 30, 40], [200, 300, 400], [3, 4, 5], [30, 40, 50], [300, 400, 500]] # values to be predicted (may be the same or different from the training input values, as long as they respect the same standard)
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
for result in results: print(result) # displays the regression result
```
```bash
{'units': 1.0, 'tens': 0.0, 'hundreds': 0.0, 'classify': 'units'}
{'units': 0.0, 'tens': 1.0, 'hundreds': 0.0, 'classify': 'tens'}
{'units': 0.0, 'tens': 0.0, 'hundreds': 1.0, 'classify': 'hundreds'}
{'units': 1.0, 'tens': 0.0, 'hundreds': 0.0, 'classify': 'units'}
{'units': 0.0, 'tens': 1.0, 'hundreds': 0.0, 'classify': 'tens'}
{'units': 0.0, 'tens': 0.0, 'hundreds': 1.0, 'classify': 'hundreds'}
```
It is also possible to transfer learning from one model to another using the "transferLearning" method. This method receives three parameters, in the first parameter we define the path to the model that will transfer the learning, in the second parameter we define the path to the model that will receive the learning and in the third parameter we define the path to the model that will be saved with the union of learning from the two previous models.
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# training phase (pattern recognition)
inputs = [[1, 2, 3], [10, 20, 30], [4, 5, 6], [40, 50, 60]] # input examples
outputs = ['units', 'tens', 'units', 'tens'] # output examples
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
sapiensnb.saveModel(path='transmitter_model') # saves the pre-trained model with the prefix "transmitter_model"
```
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# training phase (pattern recognition)
inputs = [[70, 80, 90], [700, 800, 900]] # input examples
outputs = ['tens', 'hundreds'] # output examples
sapiensnb.fit(inputs=inputs, outputs=outputs) # training for pattern recognition
sapiensnb.saveModel(path='receiver_model') # saves the pre-trained model with the prefix "receiver_model"
```
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# learning transfer phase
sapiensnb.transferLearning( # method for combining learnings
    transmitter_path='transmitter_model-88P.nb', # model that will transmit learning
    receiver_path='receiver_model-78P.nb', # model that will receive the learning
    rescue_path='complete_model' # model that will be saved with the complete union of learnings
) # the model will be saved with the name "complete_model-135P.nb"
# the final model may have fewer parameters than the sum of the previous parameters because there are some configuration parameters that will be the same and will not need repetition
```
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# prediction phase or execution phase (application of knowledge)
sapiensnb.loadModel(path='complete_model-135P.nb') # loads the pre-trained model
inputs = [[2, 3, 4], [20, 30, 40], [600, 700, 800]] # values to be predicted
results = sapiensnb.predict(inputs=inputs) # calling the prediction function that will return the results
for result in results: print(result) # displays the regression result
```
```bash
{'tens': 0.0, 'hundreds': 0.0, 'units': 1.0, 'classify': 'units'}
{'tens': 1.0, 'hundreds': 0.0, 'units': 0.0, 'classify': 'tens'}
{'tens': 0.0, 'hundreds': 1.0, 'units': 0.0, 'classify': 'hundreds'}
```
You can use the "test" function to test your model's learning. This function will return a data dictionary with the percentage of hits in the "hits" key and the percentage of errors in the "errors" key. If the level of assertiveness is not meeting your needs, you can retrain your model with a greater amount of example data and/or with a greater variability of input values.
```python
from sapiensnb import SapiensNB # module main class import
sapiensnb = SapiensNB() # class object instantiation
# test phase (testing the level of learning)
sapiensnb.loadModel(path='complete_model-135P.nb') # loads the pre-trained model
inputs = [[2, 3, 4], [20, 30, 40], [200, 300, 400], [3, 4, 5], [30, 40, 50], [300, 400, 500], [4, 5, 6], [40, 50, 60], [400, 500, 600], [5, 6, 7], [50, 60, 70], [500, 600, 700]] # values to be predicted
outputs = ['units', 'tens', 'hundreds', 'units', 'tens', 'hundreds', 'units', 'tens', 'hundreds', 'units', 'tens', 'hundreds'] # expected values as a response in prediction
results = sapiensnb.test(inputs=inputs, outputs=outputs) # function call to test learning
print(results) # displays the test result with the percentage of correct answers and the percentage of errors with values between 0 and 1 (0 for 0% and 1 for 100%)
# the acceptable level of assertiveness will depend on the project's precision needs, but for most cases we use the pareto rule (up to 20% errors are tolerable)
```
```bash
{'hits': 0.8333333333333334, 'errors': 0.16666666666666663}
```

## Methods
### fit: Train the model with input and output examples for pattern recognition
Parameters
| Name    | Description                                     | Type  | Default Value |
|---------|-------------------------------------------------|-------|---------------|
| inputs  | Input list for training with scalar or vector values | list  | []            |
| outputs | Output list for training with scalar, vector, matrix or tensor values | list  | []            |

### saveModel: Saves a file with the current model training
Parameters
| Name | Description                                       | Type | Default Value |
|------|---------------------------------------------------|------|---------------|
| path | Path with the address and file name of the model to be saved | str  | ''           |

### loadModel: Load a pre-trained model
Parameters
| Name | Description                                       | Type | Default Value |
|------|---------------------------------------------------|------|---------------|
| path | Path with the address and file name of the model to be loaded | str  | ''            |

### transferLearning: Transfer learning from one model to another
Parameters
| Name             | Description                                               | Type | Default Value |
|------------------|-----------------------------------------------------------|------|---------------|
| transmitter_path | Path with the address and file name of the model that will transfer the learning | str  | ''            |
| receiver_path    | Path with the address and file name of the model that will receive the learning | str  | ''            |
| rescue_path      | Path with address and name of the model file that will be saved with both learnings | str  | ''            |

### predict: Returns the result list with the predicted values
Parameters
| Name   | Description                                   | Type | Default Value |
|--------|-----------------------------------------------|------|---------------|
| inputs | Input list for prediction with scalar or vector values | list | []            |

### test: Tests learning by returning a dictionary with the percentage of hits and errors
Parameters
| Name   | Description                                       | Type  | Default Value |
|--------|---------------------------------------------------|-------|---------------|
| inputs | Input list for prediction with scalar or vector values | list  | []            |
| outputs| Test output list with scalar, vector, matrix or tensor values that are expected as a response | list  | []            |

Check out examples below with real data using the Titanic DataSet and Iris DataSet databases.
```bash
pip install pandas pyarrow
```
## Example with data from the Titanic DataSet
```python
from pandas import read_csv # import csv file reading module with pandas
from sapiensnb import SapiensNB # import of sapiensnb module main class
sapiensnb = SapiensNB() # class instantiation
# data preparation phase for training and testing
data_df = read_csv('titanic.csv') # reading csv file converted to dataframe
# selection of input and output lists
input_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'] # naming the columns that will make up the input list
inputs = data_df[input_columns].values.tolist() # reading input column values converted into lists
outputs = data_df['Survived'].values.tolist() # reading the column containing the output labels (the result will be a list with the label values)
# separation of data between training and testing
fit_x, fit_y = inputs[:int(len(inputs)*.8)], outputs[:int(len(outputs)*.8)] # separates 80% of input and output data for training
test_x, test_y = inputs[int(len(inputs)*.8):], outputs[int(len(outputs)*.8):] # separates the remaining 20% of data for testing
# machine learning model training phase
sapiensnb.fit(inputs=fit_x, outputs=fit_y) # performs machine learning model training
sapiensnb.saveModel() # saves the machine learning model
# machine learning model testing phase
results = sapiensnb.test(inputs=test_x, outputs=test_y) # tests the learning of the trained model
print(results) # displays percentage test results
```
```bash
{'hits': 0.6759776536312849, 'errors': 0.3240223463687151}
```
## Example with data from the Iris DataSet
```python
from pandas import read_csv # import csv file reading module with pandas
from sapiensnb import SapiensNB # import of sapiensnb module main class
sapiensnb = SapiensNB() # class instantiation
# data preparation phase for training and testing
data_df = read_csv('iris.csv') # reading csv file converted to dataframe
# selection of input and output lists
input_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'] # naming the columns that will make up the input list
inputs = data_df[input_columns].values.tolist() # reading input column values converted into lists
outputs = data_df['variety'].values.tolist() # reading the column containing the output labels (the result will be a list with the label values)
# separation of data between training and testing
fit_x, fit_y = inputs[:int(len(inputs)*.95)], outputs[:int(len(outputs)*.95)] # separates 95% of input and output data for training
test_x, test_y = inputs[int(len(inputs)*.95):], outputs[int(len(outputs)*.95):] # separates the remaining 5% of data for testing
# machine learning model training phase
sapiensnb.fit(inputs=fit_x, outputs=fit_y) # performs machine learning model training
sapiensnb.saveModel() # saves the machine learning model
# machine learning model testing phase
results = sapiensnb.test(inputs=test_x, outputs=test_y) # tests the learning of the trained model
print(results) # displays percentage test results
```
```bash
{'hits': 0.875, 'errors': 0.125}
```

## Contributing

We do not accept contributions that may result in changing the original code.

Make sure you are using the appropriate version.

## License

This is proprietary software and its alteration and/or distribution without the developer's authorization is not permitted.
