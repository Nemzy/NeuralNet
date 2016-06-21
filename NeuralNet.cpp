#include "NeuralNet.h"

//---------------------------------------------------------------Neuron---------------------------------------------------------------

Neuron::Neuron(unsigned numInputs)
{
    for(unsigned i=0; i<numInputs+1; i++)
    {
        m_weights.push_back(randomValue());
    }
}
    
double Neuron::randomValue()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);

    return dist(gen);
}

double Neuron::operator* (const std::vector<double> & inputs)
{
    double result = 0;
    
    auto weight = m_weights.cbegin();
    
    for(auto i = inputs.cbegin(); i != inputs.cend(); i++)
    {
        result += (*i) * (*weight);
        weight++;
    }
    
    result += *weight;
    
    result = sigmoid(result);
    
    m_output = result;
    
    return result;
}

double Neuron::sigmoid(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}
    
std::ostream& operator<< (std::ostream & out, const Neuron & neuron)
{
    out << "[";
    
    for(auto i = neuron.m_weights.cbegin(); i != neuron.m_weights.cend(); i++)
        out << " " << *i;
    
    out << " ]";
    
    return out;
}

//----------------------------------------------------------------NeuralNet---------------------------------------------------------------

NeuralNet::NeuralNet(const std::vector<unsigned>& topology)
{   
    for(auto i = topology.cbegin()+1; i != topology.cend(); i++)
    {   
        m_layers.push_back(Layer());
        
        for(unsigned j=0; j < *i; j++)
        {
            m_layers.back().push_back(Neuron(*(i-1)));
        }
    }
}

std::vector<double> NeuralNet::propagate(const std::vector<double>& inputs)
{
    if(inputs.size() != m_layers[0][0].m_weights.size()-1) throw std::invalid_argument("Wrong number of inputs!");
    
    return *this * inputs;
}

std::vector<double> NeuralNet::operator* (const std::vector<double> & inputs)
{
    std::vector<double> outputs = std::move(inputs);
    std::vector<double> neuronOutput;
    
    for(auto i = m_layers.begin(); i != m_layers.end(); i++)
    {
        
        for(auto j = (*i).begin(); j != (*i).end(); j++)
        {
            neuronOutput.push_back(*j * outputs);
        }
        
        outputs = std::move(neuronOutput);
        
    }
    
    return outputs;
}

void NeuralNet::backProp(const std::vector<TrainingExample> & examples, double tol, double alpha)
{
    std::vector<double> Error;
    
    do
    {
        Error.clear();
        
        //calculate error for all examples
        
        for(auto ex = examples.cbegin(); ex != examples.cend(); ex++)
        {
            std::vector<double> inputs = (*ex).m_inputs;
            
            std::vector<double> outputs = propagate(inputs);
            
            double error = 0;
            
            auto target = (*ex).m_outputs.cbegin();
            
            for(auto actual = outputs.cbegin(); actual != outputs.cend(); actual++)
            {
                error += 0.5 * std::pow((*target - *actual), 2);
                target++;
            }
            
            Error.push_back(error);
            
            //calculate deltas
            
            std::vector<double> deltaWeight;
            std::vector<double> delta;
            std::vector<double> prevDelta;
            target = (*ex).m_outputs.cbegin();
            
            //for all layers
            for(int layer = m_layers.size() - 1; layer >= 0; --layer)
            {
                prevDelta = std::move(delta);
                //for all neurons in current layer
                for(unsigned neuron = 0; neuron < m_layers[layer].size(); neuron++)
                {
                    //for output layer
                    if(layer == (int)m_layers.size()-1)
                    {
                        double d = (m_layers[layer][neuron].m_output - *target) * m_layers[layer][neuron].m_output * (1 - m_layers[layer][neuron].m_output);
                        target++;
                        delta.push_back(d);
                    }
                    else //for hidden layer
                    {   
                        double sum = 0.0;
                        int count = 0;
                        for(auto deltaIter = prevDelta.cbegin(); deltaIter != prevDelta.cend(); deltaIter++)
                        {
                            sum += *deltaIter * m_layers[layer+1][count].m_weights[neuron];
                            count++;
                        }
                        
                        double d = sum * m_layers[layer][neuron].m_output * (1 - m_layers[layer][neuron].m_output);
                        delta.push_back(d);
                    }
                    //for every weight in current neuron
                    unsigned n = m_layers[layer][neuron].m_weights.size();
                    for(unsigned weight = 0; weight < n; weight++)
                    {
                        if(layer > 0)//not first hidden
                        {
                            double w = delta.back() * (weight < n-1 ? m_layers[layer - 1][weight].m_output : 1);
                            deltaWeight.push_back(w);
                        }
                        else //first hidden
                        {
                            double w = delta.back() * (weight < n-1 ? (*ex).m_inputs[weight] : 1);
                            deltaWeight.push_back(w);
                        }
                    }
                }
            }
            
            //update weights
            auto iter = deltaWeight.cbegin();
            
            for(int layer = m_layers.size() - 1; layer >= 0; layer--)
                for(unsigned neuron = 0; neuron < m_layers[layer].size(); neuron++)
                    for(unsigned weight = 0; weight < m_layers[layer][neuron].m_weights.size(); weight++)
                        m_layers[layer][neuron].m_weights[weight] -= alpha * *iter++;
        }
    }
    while(std::accumulate(Error.cbegin(), Error.cend(), 0.0) / examples.size() > tol);
}

void NeuralNet::updateNNWeights(const std::vector<double> & new_weights)
{
    if(new_weights.size() != size()) throw std::invalid_argument("Too few of weights!");
    
    unsigned count = 0;
    
    for(unsigned i = 0; i < m_layers.size(); i++)
    {
        for(auto j = m_layers[i].begin(); j != m_layers[i].end(); j++)
        {
            for(unsigned k=0; k < (*j).m_weights.size(); k++)
                (*j).m_weights[k] = new_weights[count++];
        }
    }
}

unsigned NeuralNet::size() const
{
    unsigned count = 0;
    
    for(unsigned i = 0; i < m_layers.size(); i++)
    {
        for(auto j = m_layers[i].cbegin(); j != m_layers[i].cend(); j++)
        {
            count += (*j).m_weights.size();
        }
    }  
    
    return count;
}

std::ostream& operator<< (std::ostream& out, const NeuralNet & net)
{
    
    for(unsigned i=0; i < net.m_layers.size(); i++)
    {
        out << "########################Layer" << i << "########################" << std::endl;
        
        for(auto j = net.m_layers[i].cbegin(); j != net.m_layers[i].cend(); j++)
            out << *j << std::endl;
        
        out << "######################################################" << std::endl;
    }
    
    return out;
}

//---------------------------------------------------------------TrainingExample---------------------------------------------------------------

TrainingExample::TrainingExample(std::vector<double> && inputs, std::vector<double> && outputs)
    :m_inputs(inputs),
     m_outputs(outputs)
{}