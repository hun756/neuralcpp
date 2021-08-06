#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>

namespace Dense
{
    class DenseLayer
    {
        typedef double float_t;

    public:
        
        DenseLayer() = default;

        explicit DenseLayer(size_t inputLen, size_t outputLen) : 
            inputLen(inputLen), outputLen(outputLen),
            output(std::make_unique<float_t[]>(outputLen)),
            weight(std::make_unique<std::unique_ptr<float_t[]>[]>(outputLen))
        {
            std::random_device rd;
            std::mt19937_64 gen(rd());

            std::normal_distribution<float> dis(
              -2.0 / (inputLen + outputLen),
              2.0 / (inputLen + outputLen)  
            );

            // for (size_t i = 0; i < outputLen; ++i)
            // {
            //     output[i] = dis(gen);
            // }

            for (size_t i = 0; i < outputLen; ++i)
            {
                weight[i] = std::make_unique<float_t[]>(inputLen + 1);
                for (size_t j = 0; j < inputLen + 1; ++j)
                    weight[i][j] = dis(gen);
            }
        }

        #pragma region RueOf5
        DenseLayer(const DenseLayer& layer)
        {
            inputLen = layer.inputLen;
            outputLen = layer.outputLen;

            output = std::make_unique<float_t[]>(outputLen);
            for (size_t i = 0; i < outputLen; ++i)
                output[i] = layer.output[i];

            weight = std::make_unique<std::unique_ptr<float_t[]>[]>(outputLen);
            for (size_t i = 0; i < outputLen; ++i)
            {
                weight[i] = std::make_unique<float_t[]>(inputLen + 1);
                for (size_t j = 0; j < inputLen + 1; ++j)
                    weight[i][j] = layer.weight[i][j];
                    
            }   
        }

        DenseLayer(DenseLayer&& layer) 
        {
            inputLen = std::move(layer.inputLen);
            outputLen = std::move(layer.outputLen);

            output = std::move(layer.output);
            weight = std::move(layer.weight);
        }


        DenseLayer& operator =(const DenseLayer& layer)
        {
            if (this != &layer)
            {
                inputLen = layer.inputLen;
                outputLen = layer.outputLen;

                output = std::make_unique<float_t[]>(outputLen);
                for (size_t i = 0; i < outputLen; ++i)
                    output[i] = layer.output[i];

                weight = std::make_unique<std::unique_ptr<float_t[]>[]>(outputLen);
                for (size_t i = 0; i < outputLen; ++i)
                {
                    weight[i] = std::make_unique<float_t[]>(inputLen + 1);
                    for (size_t j = 0; j < inputLen + 1; ++j)
                        weight[i][j] = layer.weight[i][j];
                        
                }
            }
            return *this;
        }

        DenseLayer& operator =(DenseLayer&& layer) 
        {
            if (this != &layer )
            {
                inputLen = std::move(layer.inputLen);
                outputLen = std::move(layer.outputLen);

                output = std::move(layer.output);
                weight = std::move(layer.weight);
            }
            return *this;
        }
        #pragma endregion


        void setPrev(const DenseLayer& layer)
        {
            prev = std::make_unique<DenseLayer>(layer);
        }

        const std::unique_ptr<float_t[]>& forwardProp(const std::unique_ptr<float_t[]>& input)
        {
            assert(!prev || prev->outputLen == inputLen);
            auto& in = prev ? prev->forwardProp(input) : input;

            ///< [bias | weight] : out * (1 + in)
            ///< layer : out
            ///< output = bias + weight * layer
            ///< 
            auto vecMul = [](const std::unique_ptr<std::unique_ptr<float_t[]>[]>& _weight,
            const std::unique_ptr<float_t[]>& _layer,
            const size_t _inputLen, size_t idx, float_t* res)
            {
                for (size_t i = 0; i < _inputLen; ++i) 
                    *res += _weight[idx][i + 1] * _layer[i];

                ///< Nonlinearity : tanh
                *res = tanh(*res);
            };


            std::cout << "in = \n";
            for (size_t i = 0; i < outputLen; ++i)
            {
                std::cout << in[i] << '\n';
            }

            std::cout << "weight = \n";
            for (size_t i = 0; i < outputLen; ++i)
            {
                for (size_t j = 0; j < inputLen; ++j)
                {
                    std::cout << weight[i][j] << ' ';
                }
                std::cout << '\n';
            }
            std::cout << '\n';

            std::vector<std::thread> multiplicationThreads;
            multiplicationThreads.reserve(outputLen);

            for (size_t i = 0; i < outputLen; ++i)
            {
                output[i] = weight[i][0];
                multiplicationThreads.emplace_back(vecMul, std::ref(weight), std::ref(in), inputLen, i, &output[i]);
            }

            for (size_t i = 0; i < outputLen; ++i)
                multiplicationThreads[i].join();

            std::cout << "Output = \n";
            for (size_t i = 0; i < outputLen; ++i)
                std::cout << output[i] << '\n';
            
            std::cout << '\n';

            return output;
        }

    private:
        size_t inputLen,
            outputLen;
        std::unique_ptr<float_t[]> output;
        std::unique_ptr<std::unique_ptr<float_t[]>[]> weight;

        std::unique_ptr<DenseLayer> prev;
    };
} // namespace Dense


#endif