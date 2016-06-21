# NeuralNet

## Artificial Neural Net
---
Implementation of feed forward artificial neural net with back propagation algorithm for supervised learning.


###API:
1. Make new Neural Net:
  * NeuralNet(const std::vector<unsigned> & topology); 
```c++
/*makes new net with 2 input neurons,
                     4 hidden neurons,
                     6 hidden neurons(another hidden layer),
                     1 output neuron
*/
NeuralNet net({2, 4, 6, 1});
```