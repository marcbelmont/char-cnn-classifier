local test = require "test"

local examples = 0
local examples2 = 0
local CHECKPOINT = 20000
local confusion
local losses

local function train(data, opt,  dataVal)
    local model, criterion, params, gradParams = opt.model, opt.criterion, opt.params, opt.gradParams
    model:training()

    if not confusion then
        confusion = optim.ConfusionMatrix(opt.data.classes)
    end

    local shuffle
    if opt.cl.debug == 1 then
        shuffle = torch.range(1, data.targets:size(1))
    else
        shuffle = torch.randperm(data.targets:size(1))
    end

    local trainingSize = data.targets:size(1)
    local validationCorrect, validationLoss

    -- iterate over dataset
    for t = 1, trainingSize, opt.all.batchSize do
        if examples == 0 then
            losses = {}
            confusion:zero()
        end

        -- create batch
        if t + opt.all.batchSize > trainingSize then
            break
        end
        local inputs = torch.Tensor(opt.all.batchSize, data.inputs[1]:size(1))
        local targets = torch.Tensor(opt.all.batchSize)
        for i = 1, opt.all.batchSize do
            inputs[i] = data.inputs[shuffle[t + i - 1]]:clone()
            targets[i] = data.targets[shuffle[t + i - 1]]
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new params
            if x ~= params then
                params:copy(x)
            end

            -- reset gradients
            gradParams:zero()

            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)
            local err = criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            for i = 1, opt.all.batchSize do
                confusion:add(outputs[i], targets[i])
            end

            -- return f and df/dX
            return err, gradParams
        end

        -- optimize
        local _, loss = opt.optimMethod(feval, params, opt.optimState)
        table.insert(losses, loss[1])
        if opt.cl.debug ~= 1 then
            xlua.progress(t, trainingSize)
        end

        -----------------------------
        -- check on validation set --
        -----------------------------

        examples = examples + opt.all.batchSize
        if examples > CHECKPOINT then
            -- log
            confusion:__tostring__()
            if opt.cl.debug ~= 1 then
                print(confusion)
            end

            -- validation
            validationCorrect, validationLoss = test(dataVal, opt)
            model:training()
            examples = 0
        end
        examples2 = examples2 + opt.all.batchSize
        if examples2 > opt.optimState.decaySteps then
            opt.optimState.learningRate = opt.optimState.learningRate * opt.optimState.decay
            examples2 = 0
        end
    end

    -- return
    return validationLoss, validationCorrect
end

return train
