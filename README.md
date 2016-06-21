# NeuralNet

## Artificial Neural Net

Implementation of feed forward artificial neural net with back propagation algorithm for supervised learning.


###API:
+ Make new Neural Net:
  * NeuralNet(const std::vector<unsigned> & topology); 
```c++
/*makes new net with 2 input neurons,
                     4 hidden neurons,
                     6 hidden neurons(another hidden layer),
                     1 output neuron
*/
NeuralNet net({2, 4, 6, 1});
```
+ Propagate inputs:
  * std::vector<double> propagate(const std::vector<double> & inputs);
```c++
/*propagate input {0.5, 0.5} through net and get output*/
std::vector<double> output = net.propagate({.5, .5});
```
+ Training Neural Net:
  * void backProp(const std::vector<TrainingExample> & examples, double tol, double alpha);
```c++
/*vector of training examples*/
std::vector<TrainingExample> ex;

/*make new training example TrainingExample(std::vector<double> && inputs, std::vector<double> && outputs);*/
ex.push_back(TrainingExample({.5, .5}, {.5}));

/*train with accuracy of 0.000001 and speed of learning 1.0*/
net.backProp(ex, 0.000001, 1.0);
```
+ Update weights of Neural Net:
  * void updateNNWeights(const std::vector<double> & new_weights);
```c++
/*change weights of nn with vector of new_weights.
sizes must be same!
*/
net.updateNNWeights({0.5, 1.0, 0.65});
```
+ Number of weights
  * unsigned WeightsCount() const;
```c++
/*get number of weights in nn*/
unsigned n = net.WeightsCount();
```