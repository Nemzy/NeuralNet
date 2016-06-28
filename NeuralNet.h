#ifndef NEURAL_NET
#define NEURAL_NET

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <algorithm>
#include <string>

class Neuron;
class NeuralNet;
class TrainingExample;

typedef std::vector<Neuron> Layer;

//---------------------------------------------------------------Neuron---------------------------------------------------------------
class Neuron
{
public:
    
    Neuron(unsigned numInputs);
    
private:
    
    std::vector<double> m_weights;
    double m_output;
    
    static double randomValue();
    double sigmoid(double x) const;
    double operator* (const std::vector<double> & inputs);
    
    friend class NeuralNet;
    friend std::ostream& operator<< (std::ostream & out, const Neuron & neuron);
};

std::ostream& operator<< (std::ostream & out, const Neuron & neuron);


//---------------------------------------------------------------NeuralNet---------------------------------------------------------------
class NeuralNet
{
public:
    
    NeuralNet(const std::vector<unsigned> & topology);
    
    std::vector<double> propagate(const std::vector<double> & inputs);
    
    void backProp(const std::vector<TrainingExample> & examples, double tol, double alpha);
    void updateNNWeights(const std::vector<double> & new_weights);
    
    unsigned WeightsCount() const;
    
    static void serialize(const std::string & file_path, const NeuralNet & net);
    static NeuralNet deserialize(const std::string & file_path);
    
private:
    
    std::vector<Layer> m_layers;
    
    std::vector<double> operator* (const std::vector<double> & inputs);
    unsigned size() const;
    
    friend std::ostream& operator<< (std::ostream& out, const NeuralNet & net);
};

std::ostream& operator<< (std::ostream& out, const NeuralNet & net);

//---------------------------------------------------------------TrainingExample---------------------------------------------------------------
class TrainingExample
{
public:

    TrainingExample(std::vector<double> && inputs, std::vector<double> && outputs);
    
private:
    
    std::vector<double> m_inputs;
    std::vector<double> m_outputs;
    
    friend class NeuralNet;
};

#endif //NEURAL_NET