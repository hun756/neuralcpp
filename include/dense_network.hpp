#ifndef DENSE_NETWORK_HPP
#define DENSE_NETWORK_HPP

#include <memory>
#include <cassert>
#include "dense_layer.hpp"

namespace Dense
{
    class DenseNetwork
    {
        typedef double float_t;

    public:
        DenseNetwork() = default;

        explicit DenseNetwork(
            size_t layerCnt,
            size_t inputLen,
            size_t outputLen,
            size_t hiddenLen
        )   :   layerCnt(layerCnt),
                inputLen(inputLen),
                outputSize(outputLen),
                hiddenLayer(hiddenLen)
        {
            assert(layerCnt > 0);
            if (layerCnt == 1)
                outputLayer = std::make_unique<DenseLayer>(hiddenLen, outputLen);
            else
            {
                outputLayer = std::make_unique<DenseLayer>(hiddenLen, outputLen);
                auto* curLayer = &outputLayer->prev;
                for (size_t i = 0; i < layerCnt - 2; ++i)
                {
                    auto hidden_layer = std::make_unique<DenseLayer>(hiddenLen, hiddenLen);
                    *curLayer = std::move(hidden_layer);
                    curLayer = &(*curLayer)->prev;
                }

                auto firstLayer = std::make_unique<DenseLayer>(inputLen, hiddenLen);
                *curLayer = std::move(firstLayer);
            }
        }

        ~DenseNetwork() = default;

        float_t forwardPropError(const std::unique_ptr<float_t[]>& input, const std::unique_ptr<float_t[]>& expected) 
        {
            auto& result = outputLayer->forwardProp(input);
            float_t res = 0;

            // mse
            for (size_t i = 0; i < outputSize; ++i)
            {
                float_t diff = result[i] - expected[i];
                res += diff * diff;
            }

            return res;
        }

    private:
        size_t layerCnt,
            inputLen, outputSize, hiddenLayer;
        std::unique_ptr<DenseLayer> outputLayer;
        
    };
} // namespace Dense

#endif /* end of include guard :  DENSE_NETWORK_HPP */